#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal_path {
//! FLUX.2 [klein] 4B end to end through proxima: template -> Qwen3 taps -> DiT
//! Euler loop -> VAE decode -> PNG. PROXNUM-588's S8 example.
//!
//!   cargo run --release -p proxima-model-interop --features std,metal \
//!     --example flux2_klein -- --seed 42 --out image.png
//!
//! `--stage text` keeps the S4 diagnostic (it writes the three taps as raw host
//! elements); `--text-reference <dump>` replaces our text encoder's output with the
//! sd.cpp dump (raw host elements), which isolates the DiT and the VAE from the
//! text gap.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use conflaguration::Settings;
use proxima_model_interop::ServingConfig;
use proxima_model_interop::diffusion::{
    Checkpoint, Decoder, Denoiser, DiffusionSettings, DitConfig, DitInputs, Graph as ModelGraph,
    VaeConfig, attention_scale, build_dit, build_vae_decode, head_norm_count, write_png,
};
use proxima_model_interop::{
    LoadedModel, ServingSettings, architecture_from_hf_config, parse_hf_config,
};
use proxima_numeric::{FlowMatchSchedule, NumericFormat, PackedRows, Philox, euler_step};
use proxima_tensor::QuantizedBlock;
use proxima_tensor::op::NodeId;
use proxima_tensor::spec::multi_axis_rope_tables;
use proxima_tokenizer::Vocab;

const TEMPLATE_PREFIX: &str = "<|im_start|>user\n";
const TEMPLATE_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
const PROMPT: &str = "a red fox sitting in fresh snow, morning light, photograph";
const PAD_ID: u32 = 151_643;
const MIN_LENGTH: usize = 512;
const TAP_LAYERS: [usize; 3] = [8, 17, 26];
const HIDDEN: usize = 2560;
const JOINT_ATTENTION: usize = 7680;

const LATENT_SIDE: u32 = 64;
const IMAGE_TOKENS: usize = 64 * 64;
const LATENT_CHANNELS: usize = 128;
const STEPS: u32 = 4;

fn default_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join("models/image_gen/flux2-klein-4b")
}

fn reference_dir() -> PathBuf {
    std::env::var("PROXIMA_FLUX2_KLEIN_REF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_default())
                .join(".cache/kvgate/proxnum588/ref")
        })
}

fn read_vocab(dir: &Path) -> Result<Vocab, String> {
    let tokenizer = std::fs::read(dir.join("tokenizer/tokenizer.json")).map_err(|why| why.to_string())?;
    let config_bytes = std::fs::read(dir.join("text_encoder/config.json")).map_err(|why| why.to_string())?;
    let config: serde_json::Value =
        serde_json::from_slice(&config_bytes).map_err(|why| why.to_string())?;
    let number = |key: &str| {
        config
            .get(key)
            .and_then(serde_json::Value::as_u64)
            .map(|id| id as u32)
    };
    proxima_tokenizer::tokenizer_json::load_tokenizer_json(
        &tokenizer,
        number("bos_token_id"),
        number("eos_token_id"),
        None,
    )
    .map_err(|why| why.to_string())
}

/// The tokenizer's declared pad token, resolved to its one id: the reference
/// masks that token's keys in the text encoder, and the rows it leaves behind are
/// what the DiT's cross-attention reads.
fn key_pad_id(dir: &Path, vocab: &Vocab) -> Result<Option<u32>, String> {
    let bytes = std::fs::read(dir.join("tokenizer/tokenizer_config.json"))
        .map_err(|why| why.to_string())?;
    let config: serde_json::Value = serde_json::from_slice(&bytes).map_err(|why| why.to_string())?;
    let Some(pad) = config.get("pad_token").and_then(serde_json::Value::as_str) else {
        return Ok(None);
    };
    let ids = proxima_tokenizer::encode_with_bos_eos(pad, vocab, false, false)
        .map_err(|why| why.to_string())?;
    Ok((ids.len() == 1).then_some(ids[0]))
}

fn padded_ids(vocab: &Vocab) -> Result<Vec<u32>, String> {
    let prompt = format!("{TEMPLATE_PREFIX}{PROMPT}{TEMPLATE_SUFFIX}");
    let mut ids = proxima_tokenizer::encode_with_bos_eos(&prompt, vocab, false, false)
        .map_err(|why| why.to_string())?;
    if ids.len() > MIN_LENGTH {
        return Err(format!("the prompt is {} tokens", ids.len()));
    }
    ids.resize(MIN_LENGTH, PAD_ID);
    Ok(ids)
}

/// The three Qwen3 taps as one `[512, 7680]` block, tap order and row order as
/// the reference dump orders them.
fn encode_text(
    dir: &Path,
    vocab: &Vocab,
    ids: &[u32],
    key_pad: Option<u32>,
    serving: &ServingConfig,
) -> Result<PackedRows, String> {
    let config_bytes = std::fs::read(dir.join("text_encoder/config.json")).map_err(|why| why.to_string())?;
    let hf_config = parse_hf_config(&config_bytes).map_err(|why| why.to_string())?;
    let architecture = architecture_from_hf_config(&hf_config).map_err(|why| why.to_string())?;
    let path = dir.join("comfy/qwen_3_4b.safetensors");
    // Owned, not leaked: the taps are copied out into `joined` below, and the
    // 7.6 GB mapping must be gone before the DiT becomes resident, or the two
    // checkpoints' resident sets overlap for the whole image run.
    let checkpoint = Checkpoint::open(&path)?;
    let model = match key_pad {
        Some(pad) => LoadedModel::load_from_safetensors_with_key_pad(
            checkpoint.manifest(),
            checkpoint.bytes(),
            checkpoint.data_start(),
            architecture,
            vocab.clone(),
            pad,
        ),
        None => LoadedModel::load_from_safetensors(
            checkpoint.manifest(),
            checkpoint.bytes(),
            checkpoint.data_start(),
            architecture,
            vocab.clone(),
        ),
    }
    .map_err(|why| why.to_string())?;
    let roots = model.layer_residual_roots().to_vec();
    let outputs: Vec<_> = TAP_LAYERS.iter().map(|&layer| roots[layer]).collect();
    let values = model
        .forward_node_values_from_ids(ids, &outputs, serving)
        .map_err(|why| why.to_string())?;
    if values.len() != TAP_LAYERS.len() {
        return Err(String::from("the tap count does not match the layer list"));
    }
    let mut joined = Vec::with_capacity(MIN_LENGTH * JOINT_ATTENTION);
    for row in 0..MIN_LENGTH {
        for tap in &values {
            let start = row * HIDDEN;
            joined.extend_from_slice(&tap[start..start + HIDDEN]);
        }
    }
    drop(values);
    let model_started = Instant::now();
    drop(model);
    let model_millis = model_started.elapsed().as_millis();
    let unmap_started = Instant::now();
    drop(checkpoint);
    eprintln!(
        "text_drop: model {model_millis} ms, checkpoint (munmap) {} ms",
        unmap_started.elapsed().as_millis()
    );
    #[cfg(feature = "instrument")]
    eprintln!(
        "text_weight_paths: mapping_offset_uploads {}, block_offset_bound_bytes {}, misaligned_mapping_offset_copies {}, block_copied_bytes {}, block_nocopy_bound_bytes {}",
        omega::metal::MAPPING_OFFSET_UPLOADS.get(),
        omega::metal::BLOCK_OFFSET_BOUND_BYTES.get(),
        omega::metal::MISALIGNED_MAPPING_OFFSET_COPIES.get(),
        omega::metal::BLOCK_COPIED_BYTES.get(),
        omega::metal::BLOCK_NOCOPY_BOUND_BYTES.get(),
    );
    PackedRows::from_values(&NumericFormat::host_element(), &joined).map_err(|why| why.to_string())
}

/// The reference dump is channel-major (`index = token + tokens * channel`).
fn token_major<T: Copy + Default>(channel_major: &[T]) -> Vec<T> {
    let mut values = vec![T::default(); channel_major.len()];
    for channel in 0..LATENT_CHANNELS {
        for token in 0..IMAGE_TOKENS {
            values[token * LATENT_CHANNELS + channel] =
                channel_major[token + IMAGE_TOKENS * channel];
        }
    }
    values
}

/// A reference dump holds raw elements in the format the external tool writes.
fn read_dump(path: &Path, format: &NumericFormat) -> Option<PackedRows> {
    let bytes = std::fs::read(path).ok()?;
    PackedRows::from_bytes(format, bytes).ok()
}

fn fixed(value: u128, per_unit: u128, places: u32) -> String {
    let scale = 10u128.pow(places);
    let divisor = per_unit.max(1);
    let scaled = (value * scale * 2 + divisor) / (divisor * 2);
    format!("{}.{:0width$}", scaled / scale, scaled % scale, width = places as usize)
}

fn seconds(duration: Duration, places: u32) -> String {
    fixed(duration.as_nanos(), Duration::from_secs(1).as_nanos(), places)
}

#[cfg(feature = "instrument")]
fn millis(nanos: u64, places: u32) -> String {
    fixed(u128::from(nanos), Duration::from_millis(1).as_nanos(), places)
}

/// The serving settings the whole pipeline runs under: the environment's
/// `PROXIMA_SERVING_*` values, with the model directory and the token count
/// the only values this run derives.
fn serving_settings(dir: &Path, tokens: usize) -> Result<ServingSettings, String> {
    let mut settings = ServingSettings::from_env().map_err(|why| why.to_string())?;
    settings.model_path = dir.to_string_lossy().into_owned();
    settings.context_length = u32::try_from(tokens).map_err(|why| why.to_string())?;
    Ok(settings)
}

/// The per-op GPU attribution table: total, one row per `kind`, then the top
/// `top` ops by GPU span with each one's operand bytes and bytes/ns. The
/// bytes/ns column is what separates compute from traffic: an op whose rate sits
/// at the machine's streaming bandwidth is traffic-bound, one far below it is
/// leaving the dense path.
#[cfg(feature = "instrument")]
fn report_op_profile(timings: &[omega::metal::OpGpuTiming], program: &[proxima_tensor::op::Op], top: usize) {
    let total_ns: u64 = timings.iter().map(|timing| timing.gpu_ns).sum();
    let total_bytes: u64 = timings.iter().map(|timing| timing.operand_bytes).sum();
    eprintln!(
        "op_profile ops={} total_gpu_ms={} operand_bytes={total_bytes}",
        timings.len(),
        millis(total_ns, 1),
    );
    let mut by_kind: std::collections::BTreeMap<&'static str, (u64, u64, u64)> = Default::default();
    for timing in timings {
        let entry = by_kind.entry(timing.kind).or_default();
        entry.0 += 1;
        entry.1 += timing.gpu_ns;
        entry.2 += timing.operand_bytes;
    }
    for (kind, (count, ns, bytes)) in &by_kind {
        eprintln!(
            "op_profile_kind kind={kind} ops={count} gpu_ms={} share={}% operand_bytes={bytes}",
            millis(*ns, 1),
            fixed(100 * u128::from(*ns), u128::from(total_ns), 1),
        );
    }
    let mut ranked: Vec<&omega::metal::OpGpuTiming> = timings.iter().collect();
    ranked.sort_by_key(|timing| core::cmp::Reverse(timing.gpu_ns));
    for (rank, timing) in ranked.iter().take(top).enumerate() {
        let detail = match program.get(timing.node.0 as usize) {
            Some(proxima_tensor::op::Op::Reduce(reduce)) => {
                format!("keep={:?} init={:?} body={:?}", reduce.keep, reduce.init, reduce.body)
            }
            Some(proxima_tensor::op::Op::Elementwise { body, .. }) => format!("body={body:?}"),
            Some(op) => format!("{:?}", core::mem::discriminant(op)),
            None => String::from("?"),
        };
        let bytes_per_ns = if timing.gpu_ns == 0 {
            String::from("NaN")
        } else {
            fixed(u128::from(timing.operand_bytes), u128::from(timing.gpu_ns), 2)
        };
        eprintln!(
            "op_profile_top rank={} node={} kind={} extents={:?} out_axes={:?} gpu_ms={} operand_bytes={} bytes_per_ns={bytes_per_ns} weight={:?} detail={detail}",
            rank + 1,
            timing.node.0,
            timing.kind,
            timing.extents,
            timing.output_axes,
            millis(timing.gpu_ns, 1),
            timing.operand_bytes,
            timing.weight_name.as_deref().unwrap_or("-"),
        );
    }
    let accounted: u64 = ranked.iter().take(top).map(|timing| timing.gpu_ns).sum();
    eprintln!(
        "op_profile_top{top}_share={}%",
        fixed(100 * u128::from(accounted), u128::from(total_ns), 1)
    );
}

#[cfg(feature = "instrument")]
fn is_gather(op: &proxima_tensor::op::Op) -> bool {
    matches!(op, proxima_tensor::op::Op::Elementwise { operands, .. }
        if operands.iter().any(|(_, map)| matches!(map, proxima_tensor::IndexMap::Computed { .. })))
}

#[cfg(feature = "instrument")]
fn op_kind(op: &proxima_tensor::op::Op) -> String {
    match op {
        proxima_tensor::op::Op::Input { .. } => String::from("input"),
        proxima_tensor::op::Op::Reduce(reduce) => format!("reduce:{:?}", reduce.body),
        proxima_tensor::op::Op::Elementwise { .. } if is_gather(op) => String::from("gather"),
        proxima_tensor::op::Op::Elementwise { body, .. } => format!("elementwise:{body:?}"),
        _ => String::from("other"),
    }
}

/// The step class of one timed node, from its builder step and its position in
/// that step: a conv's three gathers are pad, taps and crop in program order.
#[cfg(feature = "instrument")]
fn step_class(program: &[proxima_tensor::op::Op], node: usize, step: &str, range: &core::ops::Range<usize>, timing_kind: &str) -> &'static str {
    use proxima_tensor::ScalarOp;
    use proxima_tensor::op::Op;
    let gather_ordinal = program[range.start..node].iter().filter(|op| is_gather(op)).count();
    match (step.split(':').next().unwrap_or(""), &program[node]) {
        // a gemm fused into its bias epilogue is timed at the Add node
        ("pointwise", _) if timing_kind.starts_with("reduce") => "pointwise product/reduce",
        ("resnet_add", _) => "bias add",
        ("conv", op) if is_gather(op) => ["pad-gather", "taps-gather", "crop-gather"].get(gather_ordinal).copied().unwrap_or("other"),
        ("conv", Op::Elementwise { body: ScalarOp::Select, .. }) => "pad-select",
        ("conv", Op::Elementwise { body: ScalarOp::Multiply, .. } | Op::Reduce(_)) => "conv product/reduce",
        ("conv" | "pointwise", Op::Elementwise { body: ScalarOp::Add, .. }) => "bias add",
        ("pointwise", Op::Elementwise { body: ScalarOp::Multiply, .. } | Op::Reduce(_)) => "pointwise product/reduce",
        ("group_norm", _) => "group_norm",
        ("silu", _) => "silu",
        ("upsample", _) => "upsample",
        ("attention", _) => "attention",
        _ => "other",
    }
}

/// MACs of a conv product/reduce node, `H*W*cout*cin*9`, from the builder step
/// label `conv:<tensor>:<H>x<W>:<cin>-><cout>`; a pointwise reduce's own
/// iteration extents are `rows*cout*cin`.
#[cfg(feature = "instrument")]
fn product_macs(step: &str, extents: &[u64]) -> u64 {
    let mut parts = step.split(':');
    if parts.next() != Some("conv") {
        return extents.iter().product();
    }
    let (grid, channels) = (parts.nth(1).unwrap_or(""), parts.next().unwrap_or(""));
    let numbers = |text: &str, separator: &str| -> Vec<u64> {
        text.split(separator).filter_map(|value| value.parse().ok()).collect()
    };
    match (numbers(grid, "x").as_slice(), numbers(channels, "->").as_slice()) {
        ([height, width], [cin, cout]) => height * width * cin * cout * 9,
        _ => 0,
    }
}

/// One row per timed op to `path` (node, op kind, `OpGpuTiming.kind`, extents,
/// gpu_ns, weight_name, builder step, step class, macs), then the per-class
/// table. Integer arithmetic only: ms and GMAC/s are tenths.
#[cfg(feature = "instrument")]
fn write_op_profile(
    path: &Path,
    timings: &[omega::metal::OpGpuTiming],
    graph: &ModelGraph,
) -> Result<(), String> {
    let mut rows = String::from("node\top_kind\ttiming_kind\textents\tgpu_ns\tweight_name\tstep\tclass\tmacs\n");
    let mut by_class: std::collections::BTreeMap<&'static str, (u64, u64, u64)> = Default::default();
    for timing in timings {
        let node = timing.node.0 as usize;
        let (step, range) = graph
            .steps()
            .iter()
            .find(|(_, range)| range.contains(&node))
            .map_or(("?", 0..0), |(label, range)| (label.as_str(), range.clone()));
        let class = if step == "?" { "other" } else { step_class(&graph.program, node, step, &range, timing.kind) };
        let macs = if timing.kind.starts_with("reduce") && class.ends_with("product/reduce") { product_macs(step, &timing.extents) } else { 0 };
        let entry = by_class.entry(class).or_default();
        entry.0 += 1;
        entry.1 += timing.gpu_ns;
        entry.2 += macs;
        rows.push_str(&format!(
            "{node}\t{}\t{}\t{:?}\t{}\t{}\t{step}\t{class}\t{macs}\n",
            op_kind(&graph.program[node]),
            timing.kind,
            timing.extents,
            timing.gpu_ns,
            timing.weight_name.as_deref().unwrap_or("-"),
        ));
    }
    std::fs::write(path, rows).map_err(|why| format!("{}: {why}", path.display()))?;
    let total: u64 = timings.iter().map(|timing| timing.gpu_ns).sum();
    for (class, (count, nanos, macs)) in &by_class {
        eprintln!(
            "vae_class class={class} ops={count} gpu_ns={nanos} share_permille={} macs={macs} gmac_per_s_x10={}",
            1000 * nanos / total.max(1),
            10 * macs / (*nanos).max(1),
        );
    }
    eprintln!("vae_class timed_sum_ns={total} ops={}", timings.len());
    Ok(())
}

fn dit_config() -> DitConfig {
    DitConfig {
        in_channels: 128,
        hidden: 3072,
        heads: 24,
        head_dim: 128,
        joint_attention_dim: JOINT_ATTENTION as u32,
        double_blocks: 5,
        single_blocks: 20,
        timestep_channels: 256,
        rope_theta: 2000,
    }
}

fn rope_positions<T: Copy + From<u16>>() -> Vec<T> {
    let mut positions = Vec::with_capacity((MIN_LENGTH + IMAGE_TOKENS) * 4);
    let none = T::from(0);
    for token in 0..MIN_LENGTH as u16 {
        positions.extend([none, none, none, T::from(token)]);
    }
    let width = LATENT_SIDE as u16;
    for token in 0..IMAGE_TOKENS as u16 {
        positions.extend([none, T::from(token / width), T::from(token % width), none]);
    }
    positions
}

fn vae_config(window: u32) -> Result<VaeConfig, String> {
    Ok(VaeConfig {
        window,
        channels: (LATENT_CHANNELS / 4) as u32,
        block_channels: [128, 256, 512, 512],
        layers_per_block: 2,
        groups: 32,
        out_channels: 3,
        height: LATENT_SIDE * 2,
        width: LATENT_SIDE * 2,
        conv_taps: DiffusionSettings::try_resolved()
            .map_err(|why| why.to_string())?
            .conv_taps(),
    })
}

    pub fn main() {
        #[cfg(feature = "instrument")]
        let recorder = install_telemetry();
        let outcome = run();
        #[cfg(feature = "instrument")]
        if let Some(recorder) = recorder {
            recorder.drain();
        }
        if let Err(why) = outcome {
            eprintln!("{why}");
            std::process::exit(1);
        }
    }

    /// Routes the residency pass timings (`RUST_LOG=debug`) to stdout; the
    /// ring is drained once at exit, which is enough for one short run.
    #[cfg(feature = "instrument")]
    fn install_telemetry() -> Option<std::sync::Arc<proxima_telemetry::recorder::Recorder>> {
        proxima_telemetry::emit::global::install_from_env();
        proxima_telemetry::recorder::Recorder::builder()
            .export(proxima_telemetry::export::Exporter::std())
            .and_then(|builder| builder.install())
            .ok()
    }

/// The four Euler steps: Philox noise, the DiT velocity and the flow-match
/// update. Returns the latent after step 4, the per-step seconds and the weight
/// load time.
fn step_leaves(graph: &mut ModelGraph) -> [NodeId; 3] {
    let latent = graph.host("latent", &[IMAGE_TOKENS as u32, LATENT_CHANNELS as u32], graph.unbound());
    let text = graph.host("text", &[MIN_LENGTH as u32, JOINT_ATTENTION as u32], graph.unbound());
    let timestep = graph.host("timestep", &[256], graph.unbound());
    [latent, text, timestep]
}

fn diffuse(
    dir: &Path,
    text: &PackedRows,
    seed: u64,
    serving: &ServingConfig,
    diffusion: &DiffusionSettings,
) -> Result<(PackedRows, Vec<Duration>, Duration), String> {
    let checkpoint = Checkpoint::open(&dir.join("flux-2-klein-4b.safetensors"))?;
    let activation = diffusion.dit_activation_dtype()?;
    let mut graph = ModelGraph::new(activation).map_err(|why| why.to_string())?;
    let [latent_leaf, text_leaf, timestep_leaf] = step_leaves(&mut graph);
    let (cos_bytes, sin_bytes) = multi_axis_rope_tables(
        &rope_positions(),
        &[32, 32, 32, 32],
        &[2000.0; 4],
        activation,
    )
    .map_err(|why| why.to_string())?;
    let cos_rows = PackedRows::from_bytes(graph.activation_format(), cos_bytes);
    let sin_rows = PackedRows::from_bytes(graph.activation_format(), sin_bytes);
    let rope_cos = graph.host("rope_cos", &[(MIN_LENGTH + IMAGE_TOKENS) as u32, 64], cos_rows);
    let rope_sin = graph.host("rope_sin", &[(MIN_LENGTH + IMAGE_TOKENS) as u32, 64], sin_rows);
    let config = dit_config();
    let norm_count = head_norm_count(&mut graph, &config);
    let scale = attention_scale(&mut graph, &config).map_err(|why| why.to_string())?;
    let output = build_dit(
        &mut graph,
        &config,
        &DitInputs {
            latent: latent_leaf,
            text: text_leaf,
            timestep: timestep_leaf,
            rope_cos,
            rope_sin,
            norm_count,
            attention_scale: scale,
            image_tokens: IMAGE_TOKENS as u32,
            text_tokens: MIN_LENGTH as u32,
        },
    )
    .map_err(|why| why.to_string())?;
    let rows = (MIN_LENGTH + IMAGE_TOKENS) as u64;
    let dit_load = Instant::now();
    let mut denoiser = Denoiser::new(
        &checkpoint,
        graph,
        output.velocity,
        &[rows],
        diffusion.dit_compute,
        serving,
    )?;
    let dit_seconds_load = dit_load.elapsed();
    drop(checkpoint);
    let mut latent_values = token_major(&Philox::new(seed).normal(IMAGE_TOKENS * LATENT_CHANNELS));
    let sigmas = FlowMatchSchedule::new(STEPS, IMAGE_TOKENS as u32)
        .sigmas()
        .map_err(|why| why.to_string())?;
    eprintln!(
        "dit: {} ops, {} weights, load {}s (step 1 plans)",
        denoiser.graph().program.len(),
        denoiser.graph().weights.len(),
        seconds(dit_seconds_load, 1),
    );
    let mut call_graph = ModelGraph::new(activation).map_err(|why| why.to_string())?;
    step_leaves(&mut call_graph);
    let mut dit_seconds = Vec::with_capacity(STEPS as usize);
    let mut text_values = Vec::new();
    text.decode_all(&mut text_values).map_err(|why| why.to_string())?;
    let text_rows = PackedRows::from_values(call_graph.activation_format(), &text_values)
        .map_err(|why| why.to_string())?;
    let layout = NumericFormat::host_element()
        .element_layout()
        .ok_or("the host element has no scalar layout")?;
    for step in 0..STEPS as usize {
        let settings = DiffusionSettings::try_resolved().map_err(|why| why.to_string())?;
        let format = settings.dit_activation;
        let sigma = PackedRows::from_values(&format, &[sigmas[step]]).map_err(|why| why.to_string())?;
        let timestep = settings.timestep_embedding(&sigma, 256, &format)?;
        let latent_rows = PackedRows::from_values(call_graph.activation_format(), &latent_values)
            .map_err(|why| why.to_string())?;
        let inputs = call_graph.host_blocks(&[
            ("latent", &latent_rows),
            ("text", &text_rows),
            ("timestep", &timestep),
        ])?;
        let ran = Instant::now();
        let evaluated = denoiser
            .evaluate(&inputs, &[])
            .map_err(|why| format!("dit step {}: {why}", step + 1))?;
        let elapsed = ran.elapsed();
        dit_seconds.push(elapsed);
        let velocity = evaluated
            .get(denoiser.velocity())
            .ok_or("velocity is not an output")?
            .0
            .to_vec();
        eprintln!("dit step {}: {}s", step + 1, seconds(elapsed, 2));
        euler_step(
            &mut latent_values,
            &velocity,
            sigmas[step],
            sigmas[step + 1],
            layout,
        )
        .map_err(|why| why.to_string())?;
    }
    let latent = PackedRows::from_values(&NumericFormat::host_element(), &latent_values)
        .map_err(|why| why.to_string())?;
    Ok((latent, dit_seconds, dit_seconds_load))
}

fn run() -> Result<(), String> {
    let mut dir = default_dir();
    let mut stage = String::from("image");
    let mut out: Option<PathBuf> = None;
    let mut text_reference: Option<PathBuf> = None;
    let mut seed: u64 = 42;
    let mut latent_reference: Option<PathBuf> = None;
    let mut profile_stage: Option<String> = None;
    let mut op_profile_path: Option<PathBuf> = None;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--dir" => dir = PathBuf::from(arguments.next().ok_or("--dir value")?),
            "--stage" => stage = arguments.next().ok_or("--stage value")?,
            "--out" => out = Some(PathBuf::from(arguments.next().ok_or("--out value")?)),
            "--text-reference" => {
                text_reference = Some(PathBuf::from(arguments.next().ok_or("--text-reference value")?))
            }
            "--seed" => {
                seed = arguments
                    .next()
                    .ok_or("--seed value")?
                    .parse()
                    .map_err(|why| format!("--seed: {why}"))?;
            }
            "--profile" => {
                profile_stage = Some(arguments.next().ok_or("--profile value")?)
            }
            "--vae-op-profile" => {
                op_profile_path = Some(PathBuf::from(arguments.next().ok_or("--vae-op-profile value")?))
            }
            "--latent-reference" => {
                latent_reference = Some(PathBuf::from(arguments.next().ok_or("--latent-reference value")?))
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    #[cfg(not(feature = "instrument"))]
    if profile_stage.is_some() || op_profile_path.is_some() {
        return Err(String::from(
            "--profile and --vae-op-profile need --features std,metal,instrument",
        ));
    }
    let started = Instant::now();
    let diffusion = DiffusionSettings::try_resolved().map_err(|why| why.to_string())?;
    let vocab = read_vocab(&dir)?;
    let ids = padded_ids(&vocab)?;
    let settings = serving_settings(&dir, ids.len())?;
    let serving = settings.as_serving_config(&[]);
    let key_pad = if diffusion.key_pad_mask {
        key_pad_id(&dir, &vocab)?
    } else {
        None
    };
    let host = NumericFormat::host_element();
    let text_started = Instant::now();
    let text = match &text_reference {
        Some(path) => {
            read_dump(path, &diffusion.reference_format)
                .ok_or_else(|| format!("no text dump at {}", path.display()))?
        }
        None if latent_reference.is_some() => {
            PackedRows::new(&host).map_err(|why| why.to_string())?
        }
        None => encode_text(&dir, &vocab, &ids, key_pad, &serving)?,
    };
    if latent_reference.is_none() && text.elements() != MIN_LENGTH * JOINT_ATTENTION {
        return Err(format!("text block has {} values", text.elements()));
    }
    let text_seconds = text_started.elapsed();
    eprintln!(
        "text: {} tokens, {} values in {}s ({}), pad key mask {}",
        ids.len(),
        text.elements(),
        seconds(text_seconds, 2),
        if text_reference.is_some() { "reference dump" } else { "our encoder" },
        key_pad.map_or_else(|| String::from("off"), |pad| pad.to_string())
    );
    if stage == "text" {
        if let Some(path) = out {
            std::fs::write(&path, text.as_bytes())
                .map_err(|why| format!("{}: {why}", path.display()))?;
            eprintln!("wrote {} ({} values)", path.display(), text.elements());
        }
        return Ok(());
    }

    // --- DiT: Philox noise, four Euler steps ------------------------------
    // `--latent-reference` decodes a recorded latent (the reference dump's
    // `step4.f32`) instead of running the DiT, so the VAE is the only stage
    // under measurement.
    let (latent_values, dit_seconds) = match &latent_reference {
        Some(path) => {
            let dump = read_dump(path, &diffusion.reference_format)
                .ok_or_else(|| format!("no latent dump at {}", path.display()))?;
            if dump.elements() != IMAGE_TOKENS * LATENT_CHANNELS {
                return Err(format!("latent dump has {} values", dump.elements()));
            }
            eprintln!("dit: skipped, latent from {}", path.display());
            let mut values = Vec::new();
            dump.decode_all(&mut values).map_err(|why| why.to_string())?;
            (token_major(&values), Vec::new())
        }
        None => {
            let (latent, timings, _) = diffuse(&dir, &text, seed, &serving, &diffusion)?;
            let mut values = Vec::new();
            latent.decode_all(&mut values).map_err(|why| why.to_string())?;
            (values, timings)
        }
    };

    // --- VAE: de-normalise, unpatchify, decode ---------------------------
    let vae = Checkpoint::open(&dir.join("comfy/flux2-vae.safetensors"))?;
    let (_, mean_rows) = vae.values("bn.running_mean", None, None, false)?;
    let (_, variance_rows) = vae.values("bn.running_var", None, None, false)?;
    let mut mean = Vec::new();
    mean_rows.decode_all(&mut mean).map_err(|why| why.to_string())?;
    let mut variance = Vec::new();
    variance_rows.decode_all(&mut variance).map_err(|why| why.to_string())?;
    let deviation: Vec<_> = variance.iter().map(|value| value.sqrt()).collect();
    let mut decoder_input = vec![Default::default(); IMAGE_TOKENS * 4 * (LATENT_CHANNELS / 4)];
    for row in 0..LATENT_SIDE as usize {
        for column in 0..LATENT_SIDE as usize {
            for channel in 0..LATENT_CHANNELS {
                let value = latent_values[row * LATENT_SIDE as usize * LATENT_CHANNELS
                    + column * LATENT_CHANNELS
                    + channel]
                    * deviation[channel]
                    + mean[channel];
                let merged = channel / 4;
                let offset = channel % 4;
                let target_row = row * 2 + offset / 2;
                let target_column = column * 2 + offset % 2;
                decoder_input[((target_row * LATENT_SIDE as usize * 2 + target_column)
                    * (LATENT_CHANNELS / 4))
                    + merged] = value;
            }
        }
    }
    let latent_shape = [LATENT_SIDE * LATENT_SIDE * 4, (LATENT_CHANNELS / 4) as u32, 1];
    let vae_activation = diffusion.vae_activation_dtype()?;
    let mut vae_graph = ModelGraph::new(vae_activation).map_err(|why| why.to_string())?;
    let vae_latent = vae_graph.host("latent", &latent_shape, vae_graph.unbound());
    let window = vae.window_extent("decoder.conv_in.weight")?;
    let vae_output = build_vae_decode(&mut vae_graph, &vae_config(window)?, vae_latent)
        .map_err(|why| why.to_string())?;
    let vae_load = Instant::now();
    let mut decoder = Decoder::new(
        &vae,
        vae_graph,
        vae_output.image,
        &[u64::from(latent_shape[0])],
        diffusion.vae_weights,
        &serving,
    )?;
    eprintln!(
        "vae: {} ops, {} weights, load {}s",
        decoder.graph().program.len(),
        decoder.graph().weights.len(),
        seconds(vae_load.elapsed(), 1)
    );
    let mut call_graph = ModelGraph::new(vae_activation).map_err(|why| why.to_string())?;
    call_graph.host("latent", &latent_shape, call_graph.unbound());
    let decoder_rows = PackedRows::from_values(call_graph.activation_format(), &decoder_input)
        .map_err(|why| why.to_string())?;
    let vae_blocks = call_graph.host_blocks(&[("latent", &decoder_rows)])?;
    let run_vae = |vae_inputs: &[(&str, QuantizedBlock<'_>)]| -> Result<_, String> {
    // A profiled decode runs on the decoder's profile-local runtime, so its
    // one-command-buffer-per-op plan never enters the cache a plain decode reads.
    #[cfg(feature = "instrument")]
    if profile_stage.as_deref() == Some("vae") || op_profile_path.is_some() {
        if op_profile_path.is_some() {
            let warm_started = Instant::now();
            decoder
                .evaluate_op_timed(vae_inputs)
                .map_err(|why| format!("vae warm profile: {why}"))?;
            eprintln!("vae warm decode: {:.2?} (discarded)", warm_started.elapsed());
        }
        let decode_started = Instant::now();
        let (evaluated, timings) = decoder
            .evaluate_op_timed(vae_inputs)
            .map_err(|why| format!("vae profile: {why}"))?;
        if let Some(path) = &op_profile_path {
            write_op_profile(path, &timings, decoder.graph())?;
        }
        eprintln!(
            "vae profile decode: {}s (one command buffer per op)",
            seconds(decode_started.elapsed(), 2)
        );
        report_op_profile(&timings, &decoder.graph().program, diffusion.profile_top as usize);
        // `PROXIMA_DIFFUSION_CAPTURE_NODES=<node,..>` re-runs the same plan with omega's
        // live capture on, so the record for each named op carries the launch
        // grid, the emitted entry and each operand's codec -- the facts the
        // timing table alone cannot report.
        // The slowest op and the fastest same-shaped one (a 1x1 `pointwise`,
        // whose reduce has the identical rank-5 shape) captured together: the
        // pair is what names the difference between the two kernels.
        let mut wanted_ids: Vec<u32> = Vec::new();
        wanted_ids.extend(
            diffusion
                .capture_nodes
                .split(',')
                .filter_map(|token| token.trim().parse::<u32>().ok()),
        );
        if diffusion.capture_auto {
            let mut ranked: Vec<&omega::metal::OpGpuTiming> = timings.iter().collect();
            ranked.sort_by_key(|timing| core::cmp::Reverse(timing.gpu_ns));
            wanted_ids.extend(ranked.iter().take(1).map(|timing| timing.node.0));
            wanted_ids.extend(
                timings
                    .iter()
                    .filter(|timing| {
                        timing.weight_name.as_deref().is_some_and(|name| {
                            name.ends_with(".q.weight") || name.ends_with(".proj.weight")
                        })
                    })
                    .take(1)
                    .map(|timing| timing.node.0),
            );
        }
        if !diffusion.capture_nodes.is_empty() || diffusion.capture_auto {
            let wanted = wanted_ids
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(",");
            // SAFETY: both are set before this execution starts, and the only
            // other thread is the telemetry drain pump, which never reads the
            // environment.
            unsafe {
                std::env::set_var("PROXIMA_CAPTURE_NODES", &wanted);
                std::env::set_var("PROXIMA_CAPTURE_LIVE", "1");
            }
            let _ = omega::take_captured_dispatches();
            decoder
                .evaluate_fresh(vae_inputs)
                .map_err(|why| format!("vae capture: {why}"))?;
            for record in omega::take_captured_dispatches() {
                eprintln!(
                    "vae_capture node={} kind={} entry={} extents={:?} grid={:?} operands={:?}",
                    record.node, record.kind_name, record.entry, record.extents, record.grid, record.operands
                );
            }
        }
        let image = evaluated
            .get(decoder.image())
            .ok_or("image is not an output")?
            .0
            .to_vec();
        let image_rows =
            PackedRows::from_values(&host, &image).map_err(|why| why.to_string())?;
        finish_image(&image_rows, out)?;
        return Ok(None);
    }
    let decode_started = Instant::now();
    let evaluated = decoder
        .evaluate(vae_inputs)
        .map_err(|why| format!("vae decode: {why}"))?;
    let vae_seconds = decode_started.elapsed();
    let image = evaluated.get(decoder.image()).ok_or("image is not an output")?.0;
    eprintln!("vae decode: {}s (includes the plan)", seconds(vae_seconds, 2));

    let image_rows = PackedRows::from_values(&host, image).map_err(|why| why.to_string())?;
    finish_image(&image_rows, out)?;
    Ok(Some(vae_seconds))
    };
    let vae_run = run_vae(&vae_blocks)?;
    let Some(vae_seconds) = vae_run else {
        return Ok(());
    };
    let dit_total: Duration = dit_seconds.iter().sum();
    let dit_steps = if dit_seconds.is_empty() {
        String::from("skipped")
    } else {
        dit_seconds
            .iter()
            .map(|step| seconds(*step, 1))
            .collect::<Vec<_>>()
            .join("/")
    };
    eprintln!(
        "stages: text {}s, dit {}s ({dit_steps}), vae {}s, total {}s",
        seconds(text_seconds, 2),
        seconds(dit_total, 2),
        seconds(vae_seconds, 2),
        seconds(started.elapsed(), 2)
    );
    let _ = reference_dir();
    Ok(())
}

/// Scales the raw decoder output to `[0, 1]`, writes the PNG and its raw host-element dump.
fn finish_image(image: &PackedRows, out: Option<PathBuf>) -> Result<(), String> {
    let mut values = Vec::new();
    image.decode_all(&mut values).map_err(|why| why.to_string())?;
    let pixels: Vec<_> = values
        .iter()
        .map(|value| ((value + 1.0) * 0.5).clamp(0.0, 1.0))
        .collect();
    let side = (LATENT_SIDE * 2 * 8) as usize;
    let mut rgb = Vec::with_capacity(side * side * 3);
    for token in 0..side * side {
        for channel in 0..3 {
            rgb.push(pixels[token * 3 + channel]);
        }
    }
    if let Some(path) = out {
        let settings = DiffusionSettings::try_resolved().map_err(|why| why.to_string())?;
        let unit = PackedRows::from_values(&settings.vae_activation, &rgb).map_err(|why| why.to_string())?;
        write_png(&path, side as u32, side as u32, 3, &unit, &settings.png_sample)?;
        eprintln!("wrote {} ({side}x{side})", path.display());
        let dump = path.with_extension("raw");
        let packed = PackedRows::from_values(&NumericFormat::host_element(), &pixels)
            .map_err(|why| why.to_string())?;
        std::fs::write(&dump, packed.as_bytes())
            .map_err(|why| format!("{}: {why}", dump.display()))?;
        eprintln!("wrote {}", dump.display());
    }
    Ok(())
}
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    metal_path::main();
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("the flux2-klein image path needs --features std,metal");
    std::process::exit(64);
}
