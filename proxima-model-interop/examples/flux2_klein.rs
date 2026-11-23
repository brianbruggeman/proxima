#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal_path {
//! FLUX.2 [klein] 4B end to end through proxima: template -> Qwen3 taps -> DiT
//! Euler loop -> VAE decode -> PNG. PROXNUM-588's S8 example.
//!
//!   cargo run --release -p proxima-model-interop --features std,metal \
//!     --example flux2_klein -- --seed 42 --out image.png
//!
//! `--stage text` keeps the S4 diagnostic (it writes the three taps as raw f32);
//! `--text-reference <text.f32>` replaces our text encoder's output with the
//! sd.cpp dump, which isolates the DiT and the VAE from the text gap.

use std::path::{Path, PathBuf};
use std::time::Instant;

use proxima_model_interop::ServingConfig;
use proxima_model_interop::diffusion::{
    Checkpoint, DitConfig, DitInputs, Graph as ModelGraph, VaeConfig, build_dit,
    build_vae_decode, load_weights, timestep_sinusoid, write_png,
};
use proxima_model_interop::{
    ContextLength, GPU_LAYERS_ALL, LoadedModel, architecture_from_hf_config, parse_hf_config,
};
use proxima_numeric::codec::Codec;
use proxima_numeric::{DType, FloatLayout, FlowMatchSchedule, Philox, euler_step};
use proxima_tensor::QuantizedBlock;
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

fn safetensors_data_start(file_bytes: &[u8]) -> u64 {
    let mut prefix = [0u8; 8];
    prefix.copy_from_slice(&file_bytes[..8]);
    8 + u64::from_le_bytes(prefix)
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
fn encode_text(dir: &Path, vocab: &Vocab, ids: &[u32]) -> Result<Vec<f32>, String> {
    let config_bytes = std::fs::read(dir.join("text_encoder/config.json")).map_err(|why| why.to_string())?;
    let hf_config = parse_hf_config(&config_bytes).map_err(|why| why.to_string())?;
    let architecture = architecture_from_hf_config(&hf_config).map_err(|why| why.to_string())?;
    let path = dir.join("comfy/qwen_3_4b.safetensors");
    // Owned, not leaked: the taps are copied out into `joined` below, and the
    // 7.6 GB mapping must be gone before the DiT becomes resident, or the two
    // checkpoints' resident sets overlap for the whole image run.
    let bytes = std::fs::read(&path).map_err(|why| why.to_string())?;
    let manifest = proxima_safetensors::parse_complete(&bytes).map_err(|why| why.to_string())?;
    let data_start = safetensors_data_start(&bytes);
    // `PROXIMA_FLUX2_KLEIN_NO_KEY_PAD` forces the mask off, so one binary can
    // A/B it (a mask that changes nothing is either a no-op or not in the graph).
    let key_pad = if std::env::var_os("PROXIMA_FLUX2_KLEIN_NO_KEY_PAD").is_some() {
        None
    } else {
        key_pad_id(&dir, &vocab)?
    };
    let model = match key_pad {
        Some(pad) => LoadedModel::load_from_safetensors_with_key_pad(
            &manifest,
            &bytes,
            data_start,
            architecture,
            vocab.clone(),
            pad,
        ),
        None => LoadedModel::load_from_safetensors(&manifest, &bytes, data_start, architecture, vocab.clone()),
    }
    .map_err(|why| why.to_string())?;
    let roots = model.layer_residual_roots().to_vec();
    let outputs: Vec<_> = TAP_LAYERS.iter().map(|&layer| roots[layer]).collect();
    let mut serving = ServingConfig::default();
    serving.gpu_layers = GPU_LAYERS_ALL;
    serving.context_length = ContextLength::Within(MIN_LENGTH as u32);
    serving.model_path = dir.to_str().unwrap_or_default();
    let values = model
        .forward_node_values_from_ids(ids, &outputs, &serving)
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
    drop(model);
    drop(manifest);
    drop(bytes);
    Ok(joined)
}

/// The reference dump is channel-major (`index = token + tokens * channel`).
fn token_major(channel_major: &[f32]) -> Vec<f32> {
    let mut values = vec![0.0f32; channel_major.len()];
    for channel in 0..LATENT_CHANNELS {
        for token in 0..IMAGE_TOKENS {
            values[token * LATENT_CHANNELS + channel] =
                channel_major[token + IMAGE_TOKENS * channel];
        }
    }
    values
}

fn read_dump(path: &Path) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    Some(
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect(),
    )
}

fn runtime() -> omega::config::RuntimeConfig {
    let serving = ServingConfig::default();
    omega::config::RuntimeConfig::builder()
        .with_engine("gpu")
        .with_gpu_driver("metal")
        .with_allow_fusion(true)
        .with_numeric(serving.numeric_policy)
        .with_exact_activations(serving.exact_activations)
        .build()
        .expect("metal runtime")
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
        "op_profile ops={} total_gpu_ms={:.1} operand_bytes={total_bytes}",
        timings.len(),
        total_ns as f64 / 1e6,
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
            "op_profile_kind kind={kind} ops={count} gpu_ms={:.1} share={:.1}% operand_bytes={bytes}",
            *ns as f64 / 1e6,
            100.0 * *ns as f64 / total_ns as f64,
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
            f64::NAN
        } else {
            timing.operand_bytes as f64 / timing.gpu_ns as f64
        };
        eprintln!(
            "op_profile_top rank={} node={} kind={} extents={:?} out_axes={:?} gpu_ms={:.1} operand_bytes={} bytes_per_ns={bytes_per_ns:.2} weight={:?} detail={detail}",
            rank + 1,
            timing.node.0,
            timing.kind,
            timing.extents,
            timing.output_axes,
            timing.gpu_ns as f64 / 1e6,
            timing.operand_bytes,
            timing.weight_name.as_deref().unwrap_or("-"),
        );
    }
    let accounted: u64 = ranked.iter().take(top).map(|timing| timing.gpu_ns).sum();
    eprintln!(
        "op_profile_top{top}_share={:.1}%",
        100.0 * accounted as f64 / total_ns as f64
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
fn step_class(program: &[proxima_tensor::op::Op], node: usize, step: &str, range: &core::ops::Range<usize>) -> &'static str {
    use proxima_tensor::ScalarOp;
    use proxima_tensor::op::Op;
    let gather_ordinal = program[range.start..node].iter().filter(|op| is_gather(op)).count();
    match (step.split(':').next().unwrap_or(""), &program[node]) {
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
        let class = if step == "?" { "other" } else { step_class(&graph.program, node, step, &range) };
        let reduces = matches!(graph.program[node], proxima_tensor::op::Op::Reduce(_));
        let macs = if reduces && class.ends_with("product/reduce") { product_macs(step, &timing.extents) } else { 0 };
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

fn rope_positions() -> Vec<f32> {
    let mut positions = Vec::with_capacity((MIN_LENGTH + IMAGE_TOKENS) * 4);
    for token in 0..MIN_LENGTH {
        positions.extend_from_slice(&[0.0, 0.0, 0.0, token as f32]);
    }
    let width = LATENT_SIDE as usize;
    for token in 0..IMAGE_TOKENS {
        positions.extend_from_slice(&[0.0, (token / width) as f32, (token % width) as f32, 0.0]);
    }
    positions
}

fn vae_config() -> VaeConfig {
    VaeConfig {
        channels: (LATENT_CHANNELS / 4) as u32,
        block_channels: [128, 256, 512, 512],
        layers_per_block: 2,
        groups: 32,
        out_channels: 3,
        height: LATENT_SIDE * 2,
        width: LATENT_SIDE * 2,
        conv_taps: proxima_model_interop::diffusion::DiffusionSettings::resolved().conv_taps(),
    }
}

    pub fn main() {
        if let Err(why) = run() {
            eprintln!("{why}");
            std::process::exit(1);
        }
    }

/// The four Euler steps: Philox noise, the DiT velocity and the flow-match
/// update. Returns the latent after step 4, the per-step seconds and the weight
/// load time.
fn diffuse(dir: &Path, text: &[f32], seed: u64) -> Result<(Vec<f32>, Vec<f64>, f64), String> {
    let checkpoint = Checkpoint::open(&dir.join("flux-2-klein-4b.safetensors"))?;
    let mut graph = ModelGraph::new(DType::FLOAT32);
    let latent_leaf = graph.host("latent", &[IMAGE_TOKENS as u32, LATENT_CHANNELS as u32], Vec::new());
    let text_leaf = graph.host("text", &[MIN_LENGTH as u32, JOINT_ATTENTION as u32], Vec::new());
    let timestep_leaf = graph.host("timestep", &[256], Vec::new());
    let (cos_bytes, sin_bytes) = multi_axis_rope_tables(
        &rope_positions(),
        &[32, 32, 32, 32],
        &[2000.0; 4],
        DType::FLOAT32,
    )
    .map_err(|why| why.to_string())?;
    let decode = |bytes: &[u8]| -> Vec<f32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect()
    };
    let rope_cos = graph.host("rope_cos", &[(MIN_LENGTH + IMAGE_TOKENS) as u32, 64], decode(&cos_bytes));
    let rope_sin = graph.host("rope_sin", &[(MIN_LENGTH + IMAGE_TOKENS) as u32, 64], decode(&sin_bytes));
    let output = build_dit(
        &mut graph,
        &dit_config(),
        &DitInputs {
            latent: latent_leaf,
            text: text_leaf,
            timestep: timestep_leaf,
            rope_cos,
            rope_sin,
            image_tokens: IMAGE_TOKENS as u32,
            text_tokens: MIN_LENGTH as u32,
        },
    )
    .map_err(|why| why.to_string())?;
    let dit_load = Instant::now();
    let dit_weights = load_weights(&checkpoint, &graph.weights, Codec::Float16, false)?;
    let dit_integers = graph.int_buffers();
    let dit_seconds_load = dit_load.elapsed().as_secs_f64();
    drop(checkpoint);
    let mut latent_values = token_major(&Philox::new(seed).normal(IMAGE_TOKENS * LATENT_CHANNELS));
    let cos_values = decode(&cos_bytes);
    let sin_values = decode(&sin_bytes);
    let sigmas = FlowMatchSchedule::new(STEPS, IMAGE_TOKENS as u32)
        .sigmas()
        .map_err(|why| why.to_string())?;
    let planning_timestep = timestep_sinusoid(1.0, 256);
    let mut named: Vec<(&str, QuantizedBlock)> = dit_weights.blocks();
    named.extend(graph.host_blocks(
        &[
            ("latent", latent_values.as_slice()),
            ("text", text),
            ("timestep", planning_timestep.as_slice()),
            ("rope_cos", cos_values.as_slice()),
            ("rope_sin", sin_values.as_slice()),
        ],
        &dit_integers,
    ));
    let plan_started = Instant::now();
    let (mut plan, _) = omega::backend::plan_named_with_config(
        &runtime(),
        &graph.program,
        &[(MIN_LENGTH + IMAGE_TOKENS) as u64],
        &named,
        &[output.velocity],
    )
    .map_err(|why| format!("dit plan: {why}"))?;
    eprintln!(
        "dit: {} ops, {} weights, load {dit_seconds_load:.1}s, plan {:.1}s",
        graph.program.len(),
        graph.weights.len(),
        plan_started.elapsed().as_secs_f64()
    );
    let mut dit_seconds = Vec::with_capacity(STEPS as usize);
    for step in 0..STEPS as usize {
        let timestep = timestep_sinusoid(sigmas[step], 256);
        let mut step_named: Vec<(&str, QuantizedBlock)> = dit_weights.blocks();
        step_named.extend(graph.host_blocks(
            &[
                ("latent", latent_values.as_slice()),
                ("text", text),
                ("timestep", timestep.as_slice()),
                ("rope_cos", cos_values.as_slice()),
                ("rope_sin", sin_values.as_slice()),
            ],
            &dit_integers,
        ));
        let ran = Instant::now();
        let evaluated = omega::backend::execute_plan_named(&mut plan, &step_named)
            .map_err(|why| format!("dit step {}: {why}", step + 1))?;
        let elapsed = ran.elapsed().as_secs_f64();
        dit_seconds.push(elapsed);
        let velocity = evaluated
            .get(output.velocity)
            .ok_or("velocity is not an output")?
            .0
            .to_vec();
        eprintln!("dit step {}: {elapsed:.2}s", step + 1);
        euler_step(
            &mut latent_values,
            &velocity,
            sigmas[step],
            sigmas[step + 1],
            FloatLayout::BF32,
        )
        .map_err(|why| why.to_string())?;
    }
    Ok((latent_values, dit_seconds, dit_seconds_load))
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
    let vocab = read_vocab(&dir)?;
    let ids = padded_ids(&vocab)?;
    let text_started = Instant::now();
    let text = match &text_reference {
        Some(path) => read_dump(path).ok_or_else(|| format!("no text dump at {}", path.display()))?,
        None if latent_reference.is_some() => Vec::new(),
        None => encode_text(&dir, &vocab, &ids)?,
    };
    if latent_reference.is_none() && text.len() != MIN_LENGTH * JOINT_ATTENTION {
        return Err(format!("text block has {} values", text.len()));
    }
    let text_seconds = text_started.elapsed().as_secs_f64();
    eprintln!(
        "text: {} tokens, {} values in {text_seconds:.2}s ({}), pad key mask {}",
        ids.len(),
        text.len(),
        if text_reference.is_some() { "reference dump" } else { "our encoder" },
        if std::env::var_os("PROXIMA_FLUX2_KLEIN_NO_KEY_PAD").is_some() {
            String::from("forced off")
        } else {
            key_pad_id(&dir, &vocab)?.map_or_else(|| String::from("off"), |pad| pad.to_string())
        }
    );
    if stage == "text" {
        if let Some(path) = out {
            let bytes: Vec<u8> = text.iter().flat_map(|value| value.to_le_bytes()).collect();
            std::fs::write(&path, bytes).map_err(|why| format!("{}: {why}", path.display()))?;
            eprintln!("wrote {} ({} f32)", path.display(), text.len());
        }
        return Ok(());
    }

    // --- DiT: Philox noise, four Euler steps ------------------------------
    // `--latent-reference` decodes a recorded latent (the reference dump's
    // `step4.f32`) instead of running the DiT, so the VAE is the only stage
    // under measurement.
    let (latent_values, dit_seconds) = match &latent_reference {
        Some(path) => {
            let dump =
                read_dump(path).ok_or_else(|| format!("no latent dump at {}", path.display()))?;
            if dump.len() != IMAGE_TOKENS * LATENT_CHANNELS {
                return Err(format!("latent dump has {} values", dump.len()));
            }
            eprintln!("dit: skipped, latent from {}", path.display());
            (token_major(&dump), Vec::new())
        }
        None => {
            let (latent, seconds, _) = diffuse(&dir, &text, seed)?;
            (latent, seconds)
        }
    };

    // --- VAE: de-normalise, unpatchify, decode ---------------------------
    let vae = Checkpoint::open(&dir.join("comfy/flux2-vae.safetensors"))?;
    let (_, mean) = vae.values("bn.running_mean", None, None, false)?;
    let (_, variance) = vae.values("bn.running_var", None, None, false)?;
    let deviation: Vec<f32> = variance.iter().map(|value| value.sqrt()).collect();
    let mut decoder_input = vec![0.0f32; IMAGE_TOKENS * 4 * (LATENT_CHANNELS / 4)];
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
    let mut vae_graph = ModelGraph::new(DType::FLOAT32);
    let vae_latent = vae_graph.host(
        "latent",
        &[
            LATENT_SIDE * LATENT_SIDE * 4,
            (LATENT_CHANNELS / 4) as u32,
            1,
        ],
        Vec::new(),
    );
    let vae_output = build_vae_decode(&mut vae_graph, &vae_config(), vae_latent)
        .map_err(|why| why.to_string())?;
    let vae_load = Instant::now();
    let vae_weights = load_weights(&vae, &vae_graph.weights, Codec::Float16, true)?;
    let vae_integers = vae_graph.int_buffers();
    eprintln!(
        "vae: {} ops, {} weights, load {:.1}s",
        vae_graph.program.len(),
        vae_graph.weights.len(),
        vae_load.elapsed().as_secs_f64()
    );
    let mut vae_named: Vec<(&str, QuantizedBlock)> = vae_weights.blocks();
    vae_named.extend(vae_graph.host_blocks(&[("latent", decoder_input.as_slice())], &vae_integers));
    let vae_plan = Instant::now();
    // The op-timed submission takes `omega`'s own Metal plan, so a profiling run
    // plans through the Metal entry point directly; every other run goes through
    // `omega::backend`, the surface a serving caller uses.
    let vae_runtime = runtime();
    #[cfg(feature = "instrument")]
    let profiled_plan = if profile_stage.as_deref() == Some("vae") || op_profile_path.is_some() {
        Some(
            omega::metal::plan_named_with_config(
                &vae_graph.program,
                &[(LATENT_SIDE * LATENT_SIDE * 4) as u64],
                &vae_named,
                &[vae_output.image],
                &vae_runtime,
            )
            .map_err(|why| format!("vae profile plan: {why}"))?,
        )
    } else {
        None
    };
    #[cfg(feature = "instrument")]
    if let Some(plan) = profiled_plan {
        eprintln!("vae profile plan {:.1}s", vae_plan.elapsed().as_secs_f64());
        if op_profile_path.is_some() {
            let warm_started = Instant::now();
            omega::metal::execute_plan_named_op_timed(&plan, &vae_named, None)
                .map_err(|why| format!("vae warm profile: {why}"))?;
            eprintln!("vae warm decode: {:.2?} (discarded)", warm_started.elapsed());
        }
        let decode_started = Instant::now();
        let (evaluated, timings) =
            omega::metal::execute_plan_named_op_timed(&plan, &vae_named, None)
                .map_err(|why| format!("vae profile: {why}"))?;
        if let Some(path) = &op_profile_path {
            write_op_profile(path, &timings, &vae_graph)?;
        }
        eprintln!(
            "vae profile decode: {:.2}s (one command buffer per op)",
            decode_started.elapsed().as_secs_f64()
        );
        let top = std::env::var("PROXIMA_PROFILE_TOP")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(10);
        report_op_profile(&timings, &vae_graph.program, top);
        // `PROXIMA_VAE_CAPTURE=<node,..>` re-runs the same plan with omega's
        // live capture on, so the record for each named op carries the launch
        // grid, the emitted entry and each operand's codec -- the facts the
        // timing table alone cannot report.
        // The slowest op and the fastest same-shaped one (a 1x1 `pointwise`,
        // whose reduce has the identical rank-5 shape) captured together: the
        // pair is what names the difference between the two kernels.
        let mut wanted_ids: Vec<u32> = Vec::new();
        if let Ok(list) = std::env::var("PROXIMA_VAE_CAPTURE") {
            wanted_ids.extend(list.split(',').filter_map(|token| token.trim().parse::<u32>().ok()));
        }
        if std::env::var_os("PROXIMA_VAE_CAPTURE_AUTO").is_some() {
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
        if let Ok(_) = std::env::var("PROXIMA_VAE_CAPTURE")
            .or_else(|_| std::env::var("PROXIMA_VAE_CAPTURE_AUTO"))
        {
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
            let (mut capture_plan, _) = omega::backend::plan_named_with_config(
                &runtime(),
                &vae_graph.program,
                &[(LATENT_SIDE * LATENT_SIDE * 4) as u64],
                &vae_named,
                &[vae_output.image],
            )
            .map_err(|why| format!("vae capture plan: {why}"))?;
            let _ = omega::backend::execute_plan_named(&mut capture_plan, &vae_named);
            for record in omega::take_captured_dispatches() {
                eprintln!(
                    "vae_capture node={} kind={} entry={} extents={:?} grid={:?} operands={:?}",
                    record.node, record.kind_name, record.entry, record.extents, record.grid, record.operands
                );
            }
        }
        let image = evaluated
            .get(vae_output.image)
            .ok_or("image is not an output")?
            .0
            .to_vec();
        return finish_image(image, out);
    }
    let (mut plan, _) = omega::backend::plan_named_with_config(
        &vae_runtime,
        &vae_graph.program,
        &[(LATENT_SIDE * LATENT_SIDE * 4) as u64],
        &vae_named,
        &[vae_output.image],
    )
    .map_err(|why| format!("vae plan: {why}"))?;
    eprintln!("vae plan {:.1}s", vae_plan.elapsed().as_secs_f64());
    let decode_started = Instant::now();
    let evaluated = omega::backend::execute_plan_named(&mut plan, &vae_named)
        .map_err(|why| format!("vae decode: {why}"))?;
    let vae_seconds = decode_started.elapsed().as_secs_f64();
    let image = evaluated.get(vae_output.image).ok_or("image is not an output")?.0;
    eprintln!("vae decode: {vae_seconds:.2}s");

    finish_image(image.to_vec(), out)?;
    let dit_total: f64 = dit_seconds.iter().sum();
    let dit_steps = if dit_seconds.is_empty() {
        String::from("skipped")
    } else {
        dit_seconds
            .iter()
            .map(|seconds| format!("{seconds:.1}"))
            .collect::<Vec<_>>()
            .join("/")
    };
    eprintln!(
        "stages: text {text_seconds:.2}s, dit {dit_total:.2}s ({dit_steps}), vae {vae_seconds:.2}s, total {:.2}s",
        started.elapsed().as_secs_f64()
    );
    let _ = reference_dir();
    Ok(())
}

/// Scales the raw decoder output to `[0, 1]`, writes the PNG and its f32 dump.
fn finish_image(image: Vec<f32>, out: Option<PathBuf>) -> Result<(), String> {
    let pixels: Vec<f32> = image
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
        write_png(&path, side as u32, side as u32, 3, &rgb)?;
        eprintln!("wrote {} ({side}x{side})", path.display());
        let dump = path.with_extension("f32");
        let bytes: Vec<u8> = pixels.iter().flat_map(|value| value.to_le_bytes()).collect();
        std::fs::write(&dump, bytes).map_err(|why| format!("{}: {why}", dump.display()))?;
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
