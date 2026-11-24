//! PROXNUM-588 subcomponent benchmark: every stage of the FLUX.2 klein image
//! path measured through omega on Metal with the real checkpoint weights, so a
//! stage that is slow can be named and a kernel change can be judged against a
//! number instead of a wall clock.
//!
//! Run one group (fast, the shape the kernel work iterates on):
//!
//! ```text
//! git wt cargo bench -p proxima-model-interop --features std,metal \
//!   --bench flux2_klein -- vae_conv --quick
//! ```
//!
//! Full run (`--quick`: sample size 10, short measurement windows):
//!
//! ```text
//! git wt cargo bench -p proxima-model-interop --features std,metal \
//!   --bench flux2_klein -- --quick
//! ```
//!
//! Each row prints milliseconds, GMAC/s and GB/s. A group that executes zero
//! iterations is an error and a failing exit code, never a silent pass.

use std::process::ExitCode;

#[cfg(all(feature = "metal", target_os = "macos"))]
mod bench {
    use std::cell::RefCell;
    use std::fmt::Display;
    use std::hint::black_box;
    use std::path::{Path, PathBuf};
    use std::process::ExitCode;
    use std::rc::Rc;
    use std::time::{Duration, Instant};

    use conflaguration::Settings;
    use criterion::{BenchmarkGroup, Criterion, measurement::WallTime};
    use proxima_model_interop::diffusion::{
        Checkpoint, ConvTaps, Decoder, Denoiser, DiffusionSettings, DitConfig, DitInputs,
        Graph as ModelGraph, VaeConfig, attention_scale, build_dit, build_double_block,
        build_single_block, build_vae_decode, build_window_same, head_norm_count,
    };
    use proxima_model_interop::{
        BoundWeights, LoadedModel, ServingConfig, ServingSettings, architecture_from_hf_config,
        parse_hf_config,
    };
    use proxima_numeric::{NumericError, NumericFormat, PackedRows, Philox};
    use proxima_telemetry::error;
    use proxima_tensor::op::{NodeId, Op};
    use proxima_tensor::spec::{
        group_norm, nearest_upsample_x2, non_causal_attention_heads, scalar_constant_like,
        upsample_x2_index,
    };
    use proxima_tokenizer::Vocab;

    const LATENT_SIDE: u32 = 64;
    const IMAGE_TOKENS: u32 = 64 * 64;
    const LATENT_CHANNELS: u32 = 128;
    const TEXT_TOKENS: u32 = 512;
    const JOINT_ATTENTION: u32 = 7680;
    const HIDDEN: u32 = 3072;
    const HEADS: u32 = 24;
    const HEAD_DIM: u32 = 128;
    const MIN_LENGTH: usize = 512;

    type Outcome<T> = Result<T, String>;

    fn context<E: Display>(stage: &'static str) -> impl FnOnce(E) -> String {
        move |source| format!("{stage}: {source}")
    }

    fn home() -> PathBuf {
        PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
    }

    fn model_dir() -> PathBuf {
        std::env::var_os("PROXIMA_FLUX2_KLEIN_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| home().join("models/image_gen/flux2-klein-4b"))
    }

    fn vae_path() -> PathBuf {
        model_dir().join("comfy/flux2-vae.safetensors")
    }

    fn dit_path() -> PathBuf {
        model_dir().join("flux-2-klein-4b.safetensors")
    }

    /// The serving settings every arm's stage runs under: the environment's
    /// `PROXIMA_SERVING_*` values over the defaults.
    fn serving_settings() -> Outcome<ServingSettings> {
        ServingSettings::from_env().map_err(context("serving settings"))
    }

    fn denoiser(
        checkpoint: &Checkpoint,
        graph: ModelGraph,
        output: NodeId,
        symbols: &[u64],
        config: &ServingConfig<'_>,
    ) -> Outcome<Denoiser> {
        let weights = diffusion_settings()?.dit_compute;
        Denoiser::new(checkpoint, graph, output, symbols, weights, config)
            .map_err(context("dit stage"))
    }

    fn decoder(
        checkpoint: &Checkpoint,
        graph: ModelGraph,
        output: NodeId,
        symbols: &[u64],
        config: &ServingConfig<'_>,
    ) -> Outcome<Decoder> {
        let weights = diffusion_settings()?.vae_weights;
        Decoder::new(checkpoint, graph, output, symbols, weights, config)
            .map_err(context("vae stage"))
    }

    /// A weightless program (attention, upsample) as a stage: no checkpoint
    /// leaves to bind, its host leaves resident.
    fn bare_weights() -> Outcome<BoundWeights<'static>> {
        Ok(BoundWeights::new(&[]))
    }

    /// Deterministic normal noise packed in `format`: denormals and an
    /// all-zero operand are not what the real pipeline feeds these kernels.
    fn noise_rows(
        format: &NumericFormat,
        count: usize,
        seed: u32,
    ) -> Result<PackedRows, NumericError> {
        PackedRows::from_values(format, &Philox::new(u64::from(seed)).normal(count))
    }

    /// [`noise_rows`] in the graph's activation format.
    fn seeded(graph: &ModelGraph, count: usize, seed: u32) -> Result<PackedRows, NumericError> {
        noise_rows(graph.activation_format(), count, seed)
    }

    fn diffusion_settings() -> Outcome<DiffusionSettings> {
        DiffusionSettings::try_resolved().map_err(context("diffusion settings"))
    }

    /// A graph in the backbone's configured activation element type.
    fn dit_graph() -> Outcome<ModelGraph> {
        let settings = diffusion_settings()?;
        ModelGraph::with_settings(settings.dit_activation_dtype()?, settings)
    }

    /// A graph in the decoder's configured activation element type.
    fn vae_graph() -> Outcome<ModelGraph> {
        let settings = diffusion_settings()?;
        ModelGraph::with_settings(settings.vae_activation_dtype()?, settings)
    }

    fn element_bytes(format: &NumericFormat) -> u64 {
        format.size_bytes() as u64
    }

    const SCALE: u128 = 1_000;

    fn nanos(elapsed: Duration) -> u64 {
        u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX)
    }

    fn thousandths(value: u128) -> String {
        format!("{}.{:03}", value / SCALE, value % SCALE)
    }

    fn milliseconds(elapsed_nanos: u64) -> String {
        thousandths(u128::from(elapsed_nanos) / SCALE)
    }

    /// `count` units per nanosecond is giga-units per second; printed in thousandths.
    fn giga_per_second(count: u64, elapsed: Duration) -> String {
        let scaled = u128::from(count) * SCALE;
        thousandths(scaled.checked_div(elapsed.as_nanos()).unwrap_or(0))
    }

    #[derive(Default)]
    struct Stats {
        iterations: u64,
        samples: Vec<Duration>,
    }

    struct Row {
        label: String,
        median: Duration,
        macs: u64,
        bytes: u64,
        iterations: u64,
    }

    impl Row {
        fn rates(&self) -> String {
            format!(
                "mean_ms={} gmac_s={} gb_s={}",
                milliseconds(nanos(self.median)),
                giga_per_second(self.macs, self.median),
                giga_per_second(self.bytes, self.median),
            )
        }
    }

    /// Criterion's positional filters, so a filtered run does not fail the
    /// "this bench executed" assertion on a case it was asked to skip.
    fn filters() -> Vec<String> {
        std::env::args()
            .skip(1)
            .filter(|argument| !argument.starts_with('-'))
            .collect()
    }

    fn selected(group: &str, label: &str, filters: &[String]) -> bool {
        filters.is_empty()
            || filters
                .iter()
                .any(|filter| format!("{group}/{label}").contains(filter.as_str()))
    }

    /// Runs one case, errors if it did not execute, and prints milliseconds,
    /// GMAC/s and GB/s from the measured iterations. The body runs once
    /// untimed first so its error propagates here; the timed closure returns
    /// the body's result for criterion to black-box.
    fn case(
        group: &mut BenchmarkGroup<'_, WallTime>,
        group_label: &str,
        label: &str,
        macs: u64,
        bytes: u64,
        mut body: impl FnMut() -> Outcome<()>,
    ) -> Outcome<Row> {
        let filters = filters();
        let chosen = selected(group_label, label, &filters);
        if chosen {
            body().map_err(context("untimed first run"))?;
        }
        let stats = Rc::new(RefCell::new(Stats::default()));
        let meter = Rc::clone(&stats);
        group.bench_function(label, |bencher| {
            bencher.iter(|| {
                let started = Instant::now();
                let outcome = body();
                let elapsed = started.elapsed();
                let mut slot = meter.borrow_mut();
                slot.iterations += 1;
                slot.samples.push(elapsed);
                outcome
            });
        });
        let stats = stats.borrow();
        if chosen && stats.iterations == 0 {
            return Err(format!(
                "{group_label}/{label}: criterion executed zero iterations"
            ));
        }
        // Median, not mean: the first iteration can include the pipeline's
        // one-time shader compile, and criterion's own warmup may not run a
        // whole slow iteration. The median is criterion's midpoint.
        let median = if stats.samples.is_empty() {
            Duration::ZERO
        } else {
            let mut samples = stats.samples.clone();
            samples.sort_unstable();
            let middle = samples.len() / 2;
            if samples.len().is_multiple_of(2) {
                (samples[middle - 1] + samples[middle]) / 2
            } else {
                samples[middle]
            }
        };
        let row = Row {
            label: label.to_string(),
            median,
            macs,
            bytes,
            iterations: stats.iterations,
        };
        if stats.iterations > 0 {
            eprintln!(
                "flux2_klein_bench case={group_label}/{} iters={} {}",
                row.label,
                row.iterations,
                row.rates(),
            );
        }
        Ok(row)
    }

    fn macs_of(program: &[Op], symbols: &[u64]) -> Outcome<u64> {
        let shapes = proxima_tensor::infer(program, symbols).map_err(context("program shapes"))?;
        Ok(program
            .iter()
            .filter_map(|op| match op {
                Op::Reduce(reduce) => Some(shapes.of(reduce.operand).iter().product::<u64>()),
                _ => None,
            })
            .sum())
    }

    // ---------------------------------------------------------------- text --
    fn read_vocab(dir: &Path) -> Outcome<Vocab> {
        let tokenizer =
            std::fs::read(dir.join("tokenizer/tokenizer.json")).map_err(context("tokenizer"))?;
        let config_bytes =
            std::fs::read(dir.join("text_encoder/config.json")).map_err(context("text config"))?;
        let config: serde_json::Value =
            serde_json::from_slice(&config_bytes).map_err(context("text json"))?;
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
        .map_err(context("tokenizer json"))
    }

    fn padded_ids(vocab: &Vocab) -> Outcome<Vec<u32>> {
        let prompt = "<|im_start|>user\na red fox sitting in fresh snow, morning light, photograph<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        let mut ids = proxima_tokenizer::encode_with_bos_eos(prompt, vocab, false, false)
            .map_err(context("encode"))?;
        ids.resize(MIN_LENGTH, 151_643);
        Ok(ids)
    }

    struct TextModel {
        model: LoadedModel<'static>,
        _checkpoint: &'static Checkpoint,
        roots: Vec<NodeId>,
        ids: Vec<u32>,
        serving: ServingConfig<'static>,
    }

    impl TextModel {
        fn load() -> Outcome<Self> {
            let dir = model_dir();
            let vocab = read_vocab(&dir)?;
            let ids = padded_ids(&vocab)?;
            let config_bytes =
                std::fs::read(dir.join("text_encoder/config.json")).map_err(context("config"))?;
            let hf_config = parse_hf_config(&config_bytes).map_err(context("hf config"))?;
            let architecture =
                architecture_from_hf_config(&hf_config).map_err(context("architecture"))?;
            let path = dir.join("comfy/qwen_3_4b.safetensors");
            // The text encoder is one shot per image; the bench process owns it
            // for the whole text group and leaks the mapping deliberately so
            // the model can hold a 'static borrow. The image path frees it.
            let checkpoint: &'static Checkpoint = Box::leak(Box::new(
                Checkpoint::open(&path).map_err(context("qwen checkpoint"))?,
            ));
            let model = LoadedModel::load_from_safetensors(
                checkpoint.manifest(),
                checkpoint.bytes(),
                checkpoint.data_start(),
                architecture,
                vocab.clone(),
            )
            .map_err(context("load qwen"))?;
            let roots = model.layer_residual_roots().to_vec();
            let mut settings = serving_settings()?;
            settings.model_path = dir.to_string_lossy().into_owned();
            settings.context_length = MIN_LENGTH as u32;
            // leaked with the model, so the config can hold a 'static borrow
            let settings: &'static ServingSettings = Box::leak(Box::new(settings));
            let serving = settings.as_serving_config(&[]);
            Ok(Self {
                model,
                _checkpoint: checkpoint,
                roots,
                ids,
                serving,
            })
        }
    }

    fn bench_text(c: &mut Criterion, rows: &mut Vec<Row>) -> Outcome<()> {
        let text = TextModel::load()?;
        let taps: Vec<NodeId> = [8usize, 17, 26]
            .iter()
            .map(|&layer| text.roots[layer])
            .collect();
        let one_layer = [text.roots[0]];
        let tokens = u64::from(TEXT_TOKENS);
        let element = element_bytes(&NumericFormat::host_element());

        let mut group = c.benchmark_group("text");
        group.sample_size(10);
        group.warm_up_time(Duration::from_millis(200));
        group.measurement_time(Duration::from_secs(2));
        // Qwen3 4B: hidden 2560, intermediate 9728, 32 query heads and 8 kv
        // heads of width 128 (SwiGLU MLP counted as three projections). The
        // 3-tap output stops after layer 27 (1-based), so the forward runs 27
        // of the model's 36 layers.
        let attention: u64 = 2560 * 4096 + 2 * 2560 * 1024 + 4096 * 2560;
        let mlp: u64 = 3 * 2560 * 9728;
        let full_macs = 27 * (attention + mlp) * tokens;
        let full_bytes = (3 * tokens * u64::from(JOINT_ATTENTION) + tokens * 2560 * 3) * element;
        rows.push(case(
            &mut group,
            "text",
            "full_512_encode",
            full_macs,
            full_bytes,
            || {
                let values = text
                    .model
                    .forward_node_values_from_ids(&text.ids, &taps, &text.serving)
                    .map_err(context("full text encode"))?;
                black_box(values.len());
                Ok(())
            },
        )?);
        let one_macs = (attention + mlp) * tokens;
        rows.push(case(
            &mut group,
            "text",
            "one_qwen3_layer",
            one_macs,
            (tokens * 2560 * 2) * element,
            || {
                let values = text
                    .model
                    .forward_node_values_from_ids(&text.ids, &one_layer, &text.serving)
                    .map_err(context("one text layer"))?;
                black_box(values.len());
                Ok(())
            },
        )?);
        group.finish();
        drop(text);
        Ok(())
    }

    // ----------------------------------------------------------------- dit --
    fn dit_config(double_blocks: u32, single_blocks: u32) -> DitConfig {
        DitConfig {
            in_channels: LATENT_CHANNELS,
            hidden: HIDDEN,
            heads: HEADS,
            head_dim: HEAD_DIM,
            joint_attention_dim: JOINT_ATTENTION,
            double_blocks,
            single_blocks,
            timestep_channels: 256,
            rope_theta: 2000,
        }
    }

    fn rope_hosts(graph: &mut ModelGraph) -> (NodeId, NodeId) {
        let positions = (MIN_LENGTH + IMAGE_TOKENS as usize) * 64;
        let cos = seeded(graph, positions, 7);
        let sin = seeded(graph, positions, 11);
        let cos = graph.host(
            "rope_cos",
            &[(MIN_LENGTH + IMAGE_TOKENS as usize) as u32, 64],
            cos,
        );
        let sin = graph.host(
            "rope_sin",
            &[(MIN_LENGTH + IMAGE_TOKENS as usize) as u32, 64],
            sin,
        );
        (cos, sin)
    }

    fn base_inputs(graph: &mut ModelGraph) -> Outcome<(DitInputs, NodeId, NodeId)> {
        let latent = graph.host(
            "latent",
            &[IMAGE_TOKENS, LATENT_CHANNELS],
            seeded(graph, (IMAGE_TOKENS * LATENT_CHANNELS) as usize, 3),
        );
        let text = graph.host(
            "text",
            &[TEXT_TOKENS, JOINT_ATTENTION],
            seeded(graph, (TEXT_TOKENS * JOINT_ATTENTION) as usize, 5),
        );
        let settings = diffusion_settings()?;
        let format = settings.dit_activation;
        let embedding = PackedRows::from_values(&format, &[0.5])
            .map_err(|why| why.to_string())
            .and_then(|sigma| settings.timestep_embedding(&sigma, 256, &format))?;
        let timestep = graph.host("timestep", &[256], Ok(embedding));
        let (cos, sin) = rope_hosts(graph);
        let config = dit_config(0, 0);
        let norm_count = head_norm_count(graph, &config);
        let attention_scale =
            attention_scale(graph, &config).map_err(context("attention scale"))?;
        Ok((
            DitInputs {
                latent,
                text,
                timestep,
                rope_cos: cos,
                rope_sin: sin,
                norm_count,
                attention_scale,
                image_tokens: IMAGE_TOKENS,
                text_tokens: TEXT_TOKENS,
            },
            latent,
            text,
        ))
    }

    fn bench_dit_block(
        c: &mut Criterion,
        rows: &mut Vec<Row>,
        checkpoint: &Checkpoint,
    ) -> Outcome<()> {
        let settings = serving_settings()?;
        let config = settings.as_serving_config(&[]);
        let weight = element_bytes(&diffusion_settings()?.dit_compute);
        // --- one double block ---
        {
            let mut graph = dit_graph()?;
            let element = element_bytes(graph.activation_format());
            let (inputs, _, _) = base_inputs(&mut graph)?;
            let image = graph.host(
                "block_image",
                &[IMAGE_TOKENS, HIDDEN],
                seeded(&graph, (IMAGE_TOKENS * HIDDEN) as usize, 61),
            );
            let text = graph.host(
                "block_text",
                &[TEXT_TOKENS, HIDDEN],
                seeded(&graph, (TEXT_TOKENS * HIDDEN) as usize, 67),
            );
            let mut image_mod = [NodeId(0); 6];
            let mut text_mod = [NodeId(0); 6];
            for (index, slot) in image_mod.iter_mut().enumerate() {
                *slot = graph.host(
                    &format!("image_mod{index}"),
                    &[HIDDEN],
                    seeded(&graph, HIDDEN as usize, 13 + index as u32),
                );
            }
            for (index, slot) in text_mod.iter_mut().enumerate() {
                *slot = graph.host(
                    &format!("text_mod{index}"),
                    &[HIDDEN],
                    seeded(&graph, HIDDEN as usize, 29 + index as u32),
                );
            }
            let (image, text_out) = build_double_block(
                &mut graph,
                &dit_config(1, 0),
                &inputs,
                "double_blocks.0",
                [image, text],
                [image_mod, text_mod],
            )
            .map_err(context("joint block"))?;
            let symbols = [(MIN_LENGTH + IMAGE_TOKENS as usize) as u64];
            let macs = macs_of(&graph.program, &symbols)?;
            let bytes = dit_bytes(&graph.program, &symbols, weight, element)?;
            let mut stage = denoiser(checkpoint, graph, image, &symbols, &config)?;
            let mut group = c.benchmark_group("dit");
            group.sample_size(10);
            group.warm_up_time(Duration::from_millis(100));
            group.measurement_time(Duration::from_secs(1));
            rows.push(case(
                &mut group,
                "dit",
                "one_double_block",
                macs,
                bytes,
                || {
                    black_box(stage.evaluate(&[], &[]).map_err(context("dit evaluate"))?);
                    Ok(())
                },
            )?);
            group.finish();
            let _ = text_out;
        }
        // --- one single block ---
        {
            let mut graph = dit_graph()?;
            let element = element_bytes(graph.activation_format());
            let (inputs, _, _) = base_inputs(&mut graph)?;
            let stream = graph.host(
                "stream",
                &[(MIN_LENGTH + IMAGE_TOKENS as usize) as u32, HIDDEN],
                seeded(
                    &graph,
                    (MIN_LENGTH + IMAGE_TOKENS as usize) * HIDDEN as usize,
                    17,
                ),
            );
            let mut modulation = [NodeId(0); 3];
            for (index, slot) in modulation.iter_mut().enumerate() {
                *slot = graph.host(
                    &format!("single_mod{index}"),
                    &[HIDDEN],
                    seeded(&graph, HIDDEN as usize, 41 + index as u32),
                );
            }
            let output = build_single_block(
                &mut graph,
                &dit_config(0, 1),
                &inputs,
                "single_blocks.0",
                stream,
                modulation,
            )
            .map_err(context("single block"))?;
            let symbols = [(MIN_LENGTH + IMAGE_TOKENS as usize) as u64];
            let macs = macs_of(&graph.program, &symbols)?;
            let bytes = dit_bytes(&graph.program, &symbols, weight, element)?;
            let mut stage = denoiser(checkpoint, graph, output, &symbols, &config)?;
            let mut group = c.benchmark_group("dit");
            group.sample_size(10);
            group.warm_up_time(Duration::from_millis(100));
            group.measurement_time(Duration::from_secs(1));
            rows.push(case(
                &mut group,
                "dit",
                "one_single_block",
                macs,
                bytes,
                || {
                    black_box(stage.evaluate(&[], &[]).map_err(context("dit evaluate"))?);
                    Ok(())
                },
            )?);
            group.finish();
        }
        Ok(())
    }

    fn dit_bytes(
        program: &[Op],
        symbols: &[u64],
        weight_bytes: u64,
        activation_bytes: u64,
    ) -> Outcome<u64> {
        let shapes = proxima_tensor::infer(program, symbols).map_err(context("program shapes"))?;
        let count = |node: NodeId| -> u64 { shapes.of(node).iter().product::<u64>() };
        let mut bytes = 0;
        for (index, op) in program.iter().enumerate() {
            if let Op::Input { name, .. } = op {
                let element = if name.as_deref().is_some_and(|name| name.contains(".weight")) {
                    weight_bytes
                } else {
                    activation_bytes
                };
                bytes += count(NodeId(index as u32)) * element;
            }
        }
        if let Some(last) = program.len().checked_sub(1) {
            bytes += count(NodeId(last as u32)) * activation_bytes;
        }
        Ok(bytes)
    }

    fn bench_dit_full(
        c: &mut Criterion,
        rows: &mut Vec<Row>,
        checkpoint: &Checkpoint,
    ) -> Outcome<()> {
        let mut graph = dit_graph()?;
        let element = element_bytes(graph.activation_format());
        let weight = element_bytes(&diffusion_settings()?.dit_compute);
        let (inputs, _, _) = base_inputs(&mut graph)?;
        let output =
            build_dit(&mut graph, &dit_config(5, 20), &inputs).map_err(context("full dit"))?;
        let settings = serving_settings()?;
        let config = settings.as_serving_config(&[]);
        let symbols = [(MIN_LENGTH + IMAGE_TOKENS as usize) as u64];
        let macs = macs_of(&graph.program, &symbols)?;
        let bytes = dit_bytes(&graph.program, &symbols, weight, element)?;
        let mut stage = denoiser(checkpoint, graph, output.velocity, &symbols, &config)?;
        let mut group = c.benchmark_group("dit");
        group.sample_size(10);
        group.warm_up_time(Duration::from_millis(200));
        group.measurement_time(Duration::from_secs(2));
        rows.push(case(
            &mut group,
            "dit",
            "full_velocity_step",
            macs,
            bytes,
            || {
                black_box(stage.evaluate(&[], &[]).map_err(context("dit evaluate"))?);
                Ok(())
            },
        )?);
        group.finish();
        Ok(())
    }

    fn attention_program(
        graph: &mut ModelGraph,
        tokens: u32,
        heads: u32,
        head_dim: u32,
        seed: u32,
    ) -> Outcome<NodeId> {
        let half = head_dim / 2;
        let mut host = |name: &str, shape: &[u32], salt: u32| {
            let count: u32 = shape.iter().product();
            let values = seeded(graph, count as usize, seed + salt);
            graph.host(name, shape, values)
        };
        let q_even = host("q_even", &[tokens, heads, half], 1);
        let q_odd = host("q_odd", &[tokens, heads, half], 2);
        let k_even = host("k_even", &[tokens, heads, half], 3);
        let k_odd = host("k_odd", &[tokens, heads, half], 4);
        let v = host("v", &[tokens, heads, head_dim], 5);
        let config = DitConfig {
            head_dim,
            ..dit_config(0, 0)
        };
        let scale = attention_scale(graph, &config).map_err(context("attention scale"))?;
        non_causal_attention_heads(
            &mut graph.program,
            &[(q_even, k_even), (q_odd, k_odd)],
            v,
            scale,
            "u",
            "p",
            "d",
        )
        .map_err(context("attention"))
    }

    fn bench_dit_attention(c: &mut Criterion, rows: &mut Vec<Row>) -> Outcome<()> {
        let tokens = MIN_LENGTH as u32 + IMAGE_TOKENS;
        let mut graph = dit_graph()?;
        let element = element_bytes(graph.activation_format());
        let output = attention_program(&mut graph, tokens, HEADS, HEAD_DIM, 53)?;
        let symbols = [u64::from(tokens)];
        let macs = macs_of(&graph.program, &symbols)?;
        let row_count = u64::from(tokens) * u64::from(HEADS);
        let bytes = (row_count * u64::from(HEAD_DIM) * 2 + row_count * u64::from(HEAD_DIM / 2) * 4)
            * element;
        let settings = serving_settings()?;
        let config = settings.as_serving_config(&[]);
        let mut stage = Denoiser::with_weights(graph, output, &symbols, bare_weights()?, &config)
            .map_err(context("attention stage"))?;
        let mut group = c.benchmark_group("dit");
        group.sample_size(10);
        group.warm_up_time(Duration::from_millis(100));
        group.measurement_time(Duration::from_secs(1));
        rows.push(case(
            &mut group,
            "dit",
            "attention_4608_score_mix",
            macs,
            bytes,
            || {
                black_box(
                    stage
                        .evaluate(&[], &[])
                        .map_err(context("attention evaluate"))?,
                );
                Ok(())
            },
        )?);
        group.finish();
        Ok(())
    }

    // ----------------------------------------------------------------- vae --
    fn vae_config(window: u32) -> Outcome<VaeConfig> {
        Ok(VaeConfig {
            window,
            channels: LATENT_CHANNELS / 4,
            block_channels: [128, 256, 512, 512],
            layers_per_block: 2,
            groups: 32,
            out_channels: 3,
            height: LATENT_SIDE * 2,
            width: LATENT_SIDE * 2,
            conv_taps: diffusion_settings()?.conv_taps(),
        })
    }

    struct ConvCase {
        label: &'static str,
        tensor: &'static str,
        side: u32,
        in_channels: u32,
        out_channels: u32,
        groups: u32,
    }

    const CONV_CASES: [ConvCase; 4] = [
        ConvCase {
            label: "window_same_128sq_32to512",
            tensor: "decoder.conv_in",
            side: 128,
            in_channels: 32,
            out_channels: 512,
            groups: 32,
        },
        ConvCase {
            label: "window_same_256sq_512to512",
            tensor: "decoder.up.2.block.0.conv2",
            side: 256,
            in_channels: 512,
            out_channels: 512,
            groups: 32,
        },
        ConvCase {
            label: "window_same_512sq_256to256",
            tensor: "decoder.up.1.block.0.conv2",
            side: 512,
            in_channels: 256,
            out_channels: 256,
            groups: 32,
        },
        ConvCase {
            label: "window_same_1024sq_128to128",
            tensor: "decoder.up.0.block.1.conv2",
            side: 1024,
            in_channels: 128,
            out_channels: 128,
            groups: 32,
        },
    ];

    fn case_metrics(case: &ConvCase, weight_bytes: u64, activation_bytes: u64) -> (u64, u64) {
        let rows = u64::from(case.side) * u64::from(case.side);
        let macs = u64::from(case.out_channels) * rows * 9 * u64::from(case.in_channels);
        let bytes = u64::from(case.out_channels) * u64::from(case.in_channels) * 9 * weight_bytes
            + rows * u64::from(case.in_channels) * activation_bytes
            + rows * u64::from(case.out_channels) * activation_bytes;
        (macs, bytes)
    }

    /// The VAE's own factored 3x3 conv for `case`, exactly as the decoder
    /// builds it, over a pseudo-random activation leaf.
    fn conv_graph(case: &ConvCase, checkpoint: &Checkpoint) -> Outcome<(ModelGraph, NodeId)> {
        let window = checkpoint
            .window_extent(&format!("{}.weight", case.tensor))
            .map_err(context("conv window"))?;
        let config = VaeConfig {
            conv_taps: ConvTaps::Three,
            ..vae_config(window)?
        };
        let rows_n = case.side * case.side;
        let mut graph = vae_graph()?;
        let input = graph.host(
            "activation",
            &[rows_n, case.groups, case.in_channels / case.groups],
            seeded(
                &graph,
                (rows_n * case.in_channels) as usize,
                101 + case.side,
            ),
        );
        let output = build_window_same(
            &mut graph,
            input,
            case.tensor,
            (case.side, case.side),
            (case.in_channels, case.out_channels),
            (case.groups, case.groups),
            &config,
        )
        .map_err(context("conv"))?;
        Ok((graph, output))
    }

    fn bench_vae_conv(
        c: &mut Criterion,
        rows: &mut Vec<Row>,
        checkpoint: &Checkpoint,
    ) -> Outcome<()> {
        let settings = serving_settings()?;
        let config = settings.as_serving_config(&[]);
        let weight = element_bytes(&diffusion_settings()?.vae_weights);
        for case_spec in &CONV_CASES {
            let rows_n = case_spec.side * case_spec.side;
            let (graph, output) = conv_graph(case_spec, checkpoint)?;
            let symbols = [u64::from(rows_n)];
            let (macs, bytes) =
                case_metrics(case_spec, weight, element_bytes(graph.activation_format()));
            let mut stage = decoder(checkpoint, graph, output, &symbols, &config)?;
            let mut group = c.benchmark_group("vae_conv");
            group.sample_size(10);
            group.warm_up_time(Duration::from_millis(100));
            group.measurement_time(Duration::from_secs(1));
            rows.push(case(
                &mut group,
                "vae_conv",
                case_spec.label,
                macs,
                bytes,
                || {
                    black_box(stage.evaluate(&[]).map_err(context("conv evaluate"))?);
                    Ok(())
                },
            )?);
            group.finish();
        }
        Ok(())
    }

    fn bench_vae_norm_upsample(
        c: &mut Criterion,
        rows: &mut Vec<Row>,
        checkpoint: &Checkpoint,
    ) -> Outcome<()> {
        let settings = serving_settings()?;
        let config = settings.as_serving_config(&[]);
        // group_norm: the mid block's 512 channels at 128x128, 32 groups.
        {
            let mut graph = vae_graph()?;
            let element = element_bytes(graph.activation_format());
            let rows_n = 128u32 * 128u32;
            let channels = 512u32;
            let groups = 32u32;
            let x = graph.host(
                "activation",
                &[rows_n, groups, channels / groups],
                seeded(&graph, (rows_n * channels) as usize, 201),
            );
            let weight = graph.weight(
                "decoder.mid.block_1.norm1.weight",
                &[groups, channels / groups],
                None,
            );
            let bias = graph.weight(
                "decoder.mid.block_1.norm1.bias",
                &[groups, channels / groups],
                None,
            );
            let eps = scalar_constant_like(&mut graph.program, 1e-6, x)
                .map_err(context("group norm eps"))?;
            let output = group_norm(
                &mut graph.program,
                x,
                weight,
                bias,
                rows_n * (channels / groups),
                eps,
            )
            .map_err(context("group norm"))?;
            let symbols = [u64::from(rows_n)];
            let bytes = u64::from(rows_n) * u64::from(channels) * 2 * element;
            let mut stage = decoder(checkpoint, graph, output, &symbols, &config)?;
            let mut group = c.benchmark_group("vae");
            group.sample_size(10);
            group.warm_up_time(Duration::from_millis(100));
            group.measurement_time(Duration::from_secs(1));
            rows.push(case(
                &mut group,
                "vae",
                "group_norm_128sq",
                0,
                bytes,
                || {
                    black_box(stage.evaluate(&[]).map_err(context("norm evaluate"))?);
                    Ok(())
                },
            )?);
            group.finish();
        }
        // nearest_x2: the 128x128 -> 256x256 upsample that precedes a conv.
        {
            let mut graph = vae_graph()?;
            let element = element_bytes(graph.activation_format());
            let height = 128u32;
            let width = 128u32;
            let channels = 512u32;
            let groups = 32u32;
            let x = graph.host(
                "activation",
                &[height * width, groups, channels / groups],
                seeded(&graph, (height * width * channels) as usize, 211),
            );
            let index = graph.index_leaf("upsample.index", upsample_x2_index(height, width));
            let output =
                nearest_upsample_x2(&mut graph.program, x, index).map_err(context("upsample"))?;
            let symbols = [u64::from(height * width)];
            let bytes = u64::from(height * width * channels) * element * 2;
            let mut stage = Decoder::with_weights(graph, output, &symbols, bare_weights()?, &config)
                .map_err(context("upsample stage"))?;
            let mut group = c.benchmark_group("vae");
            group.sample_size(10);
            group.warm_up_time(Duration::from_millis(100));
            group.measurement_time(Duration::from_secs(1));
            rows.push(case(
                &mut group,
                "vae",
                "nearest_x2_128sq",
                0,
                bytes,
                || {
                    black_box(stage.evaluate(&[]).map_err(context("upsample evaluate"))?);
                    Ok(())
                },
            )?);
            group.finish();
        }
        Ok(())
    }

    fn bench_vae_attention(c: &mut Criterion, rows: &mut Vec<Row>) -> Outcome<()> {
        let tokens = 128u32 * 128u32;
        let mut graph = vae_graph()?;
        let element = element_bytes(graph.activation_format());
        // One head of width 512 split as 32 channel groups of 16; the same
        // shape the decoder's mid attention builds from its 1x1 projections.
        let output = attention_program(&mut graph, tokens, 32, 32, 71)?;
        let symbols = [u64::from(tokens)];
        let macs = macs_of(&graph.program, &symbols)?;
        let bytes = u64::from(tokens) * 512 * 2 * element;
        let settings = serving_settings()?;
        let config = settings.as_serving_config(&[]);
        let mut stage = Decoder::with_weights(graph, output, &symbols, bare_weights()?, &config)
            .map_err(context("attention stage"))?;
        let mut group = c.benchmark_group("vae");
        group.sample_size(10);
        group.warm_up_time(Duration::from_millis(100));
        group.measurement_time(Duration::from_secs(1));
        rows.push(case(
            &mut group,
            "vae",
            "mid_attention_16384",
            macs,
            bytes,
            || {
                black_box(stage.evaluate(&[]).map_err(context("attention evaluate"))?);
                Ok(())
            },
        )?);
        group.finish();
        Ok(())
    }

    fn bench_vae_full(
        c: &mut Criterion,
        rows: &mut Vec<Row>,
        checkpoint: &Checkpoint,
    ) -> Outcome<()> {
        let window = checkpoint
            .window_extent("decoder.conv_in.weight")
            .map_err(context("vae window"))?;
        let config = vae_config(window)?;
        let mut graph = vae_graph()?;
        let tokens = LATENT_SIDE * 2 * LATENT_SIDE * 2;
        let input = graph.host(
            "latent",
            &[tokens, LATENT_CHANNELS / 4, 1],
            seeded(&graph, (tokens * LATENT_CHANNELS / 4) as usize, 301),
        );
        let output = build_vae_decode(&mut graph, &config, input).map_err(context("vae decode"))?;
        let settings = serving_settings()?;
        let config = settings.as_serving_config(&[]);
        let symbols = [u64::from(tokens)];
        let macs = macs_of(&graph.program, &symbols)?;
        let bytes = 0;
        let mut stage = decoder(checkpoint, graph, output.image, &symbols, &config)?;
        let mut group = c.benchmark_group("vae");
        group.sample_size(10);
        group.warm_up_time(Duration::from_millis(200));
        group.measurement_time(Duration::from_secs(2));
        rows.push(case(&mut group, "vae", "full_decode", macs, bytes, || {
            black_box(stage.evaluate(&[]).map_err(context("decode evaluate"))?);
            Ok(())
        })?);
        group.finish();
        Ok(())
    }

    /// PROXNUM-687 S0b: the VAE's 3x3 conv at one shape (256x256, 512 to 512
    /// channels, 157 GMAC), isolated three ways. (a) is the VAE's own factored
    /// form, (b) the same MACs as a plain 2-axis `linear` over a staged
    /// `[M, 9 * 512]` activation, (d) a DiT-shaped `linear` of equal MACs.
    /// The tile layout is read from `OMEGA_DENSE_MMA_INPUT_LAYOUT` through
    /// `RuntimeConfig::from_env`, so one invocation measures the derived
    /// default and a second, with the variable set, measures the explicit
    /// narrower tile. Every arm reports the GEMM op's own GPU span from the
    /// op timer (one command buffer per op), its kernel kind, the plan's
    /// kernel identities, and the untimed plan wall.
    #[cfg(feature = "instrument")]
    mod isolate {
        use std::hint::black_box;
        use std::time::Instant;

        use omega::config::{DenseMmaInputLayout, RuntimeConfig};
        use omega::metal::OpGpuTiming;
        use proxima_model_interop::diffusion::{
            Checkpoint, Decoder, DiffusionSettings, Graph as ModelGraph,
        };
        use proxima_model_interop::{BoundWeights, ServingConfig};
        use proxima_numeric::DType;
        use proxima_numeric::codec::Codec;
        use proxima_tensor::spec::project_rows;

        use super::{
            CONV_CASES, ConvCase, Outcome, context, conv_graph, milliseconds, noise_rows, seeded,
            serving_settings,
        };

        const TAPS: u32 = 9;
        const CONTROL_ROWS: u32 = 4096;
        const CONTROL_DEPTH: u32 = 3072;
        // 4096 * 3072 * 12480 = 157.04 GMAC, the conv's 66564 * 4608 * 512.
        const CONTROL_COLUMNS: u32 = 12480;
        const WARMUP: usize = 2;
        const ITERATIONS: usize = 10;

        /// Runs `body` under the serving config with `tiles` attached as the
        /// omega runtime: the tile layout A/B is a config choice, not a second
        /// planning path.
        fn with_tiles<R>(
            tiles: DenseMmaInputLayout,
            body: impl FnOnce(&ServingConfig<'_>) -> Outcome<R>,
        ) -> Outcome<R> {
            let settings = serving_settings()?;
            let base = settings.as_serving_config(&[]);
            let runtime = RuntimeConfig::builder()
                .with_engine("gpu")
                .with_gpu_driver("metal")
                .with_allow_fusion(true)
                .with_numeric(base.numeric_policy)
                .with_exact_activations(base.exact_activations)
                .with_dense_mma_input_layout(tiles)
                .build()
                .map_err(context("metal runtime"))?;
            body(&base.with_omega_runtime(&runtime))
        }

        struct Arm<'a> {
            label: &'static str,
            stage: &'a mut Decoder,
            macs: u64,
        }

        fn padded_rows(case: &ConvCase) -> u32 {
            (case.side + 2) * (case.side + 2)
        }

        fn conv_macs(case: &ConvCase) -> u64 {
            u64::from(padded_rows(case))
                * u64::from(TAPS * case.in_channels)
                * u64::from(case.out_channels)
        }

        fn median(sorted: &[u64]) -> u64 {
            let middle = sorted.len() / 2;
            if sorted.len().is_multiple_of(2) {
                (sorted[middle - 1] + sorted[middle]) / 2
            } else {
                sorted[middle]
            }
        }

        fn nearest_rank_p90(sorted: &[u64]) -> u64 {
            sorted[(sorted.len() * 9).div_ceil(10) - 1]
        }

        fn cov_permille(samples: &[u64]) -> u64 {
            let count = samples.len() as u64;
            let mean = samples.iter().sum::<u64>() / count;
            let squares: u128 = samples
                .iter()
                .map(|&sample| {
                    let delta = i128::from(sample) - i128::from(mean);
                    (delta * delta) as u128
                })
                .sum();
            let stdev = (squares / u128::from(count)).isqrt() as u64;
            stdev * 1000 / mean
        }

        fn heaviest_reduce(timings: &[OpGpuTiming]) -> Outcome<&OpGpuTiming> {
            timings
                .iter()
                .filter(|timing| timing.kind.starts_with("reduce"))
                .max_by_key(|timing| timing.gpu_ns)
                .ok_or_else(|| String::from("the op timer reported no reduce op"))
        }

        fn report_ops(label: &str, timings: &[OpGpuTiming]) {
            for timing in timings {
                eprintln!(
                    "s0b_op label={label} node={} kind={} variant={} extents={:?} out_axes={:?} operands={} gpu_ms={}",
                    timing.node.0,
                    timing.kind,
                    timing.packed_kernel_variant,
                    timing.extents,
                    timing.output_axes,
                    timing.operand_count,
                    milliseconds(timing.gpu_ns),
                );
            }
        }

        fn measure(arm: &mut Arm<'_>, tiles: DenseMmaInputLayout) -> Outcome<()> {
            let tile_label = if tiles.derive_from_operands {
                "derived"
            } else {
                "explicit"
            };
            let label = format!("{}__{tile_label}", arm.label);
            let mut gemm_ns = Vec::new();
            let mut wall_ns = Vec::new();
            let mut timed_wall_ns = Vec::new();
            let mut gemm_kind = "";
            for pass in 0..WARMUP + ITERATIONS {
                let started = Instant::now();
                black_box(arm.stage.evaluate(&[]).map_err(context("stage evaluate"))?);
                let wall = started.elapsed().as_nanos() as u64;
                let timed_started = Instant::now();
                let (evaluated, timings) = arm
                    .stage
                    .evaluate_op_timed(&[])
                    .map_err(context("stage op-timed evaluate"))?;
                let timed_wall = timed_started.elapsed().as_nanos() as u64;
                black_box(evaluated);
                if pass < WARMUP {
                    continue;
                }
                if pass == WARMUP {
                    report_ops(&label, &timings);
                }
                let heaviest = heaviest_reduce(&timings)?;
                gemm_kind = heaviest.kind;
                gemm_ns.push(heaviest.gpu_ns);
                wall_ns.push(wall);
                timed_wall_ns.push(timed_wall);
            }
            if gemm_ns.len() != ITERATIONS {
                return Err(format!(
                    "{label}: measured {} of {ITERATIONS} iterations",
                    gemm_ns.len()
                ));
            }
            let cov = cov_permille(&gemm_ns);
            gemm_ns.sort_unstable();
            wall_ns.sort_unstable();
            timed_wall_ns.sort_unstable();
            let gemm_median = median(&gemm_ns);
            eprintln!(
                "s0b_arm label={label} kind={gemm_kind} macs={} n={ITERATIONS} gemm_ms_median={} gemm_ms_p90={} gmac_s_x10_median={} cov_permille={cov} wall_ms_median={} timed_wall_ms_median={}",
                arm.macs,
                milliseconds(gemm_median),
                milliseconds(nearest_rank_p90(&gemm_ns)),
                arm.macs * 10 / gemm_median,
                milliseconds(median(&wall_ns)),
                milliseconds(median(&timed_wall_ns)),
            );
            Ok(())
        }

        fn form_a(
            checkpoint: &Checkpoint,
            case: &ConvCase,
            tiles: DenseMmaInputLayout,
        ) -> Outcome<(DType, Codec)> {
            let (graph, output) = conv_graph(case, checkpoint)?;
            let activation = graph.activation();
            let codec = DiffusionSettings::try_resolved()
                .map_err(context("diffusion settings"))?
                .vae_weights;
            with_tiles(tiles, |config| {
                let mut stage = Decoder::new(
                    checkpoint,
                    graph,
                    output,
                    &[u64::from(case.side * case.side)],
                    codec,
                    config,
                )
                .map_err(context("conv stage"))?;
                measure(
                    &mut Arm {
                        label: "a_factored_conv",
                        stage: &mut stage,
                        macs: conv_macs(case),
                    },
                    tiles,
                )
            })?;
            Ok((activation, codec))
        }

        fn form_b(
            checkpoint: &Checkpoint,
            case: &ConvCase,
            activation: DType,
            tiles: DenseMmaInputLayout,
        ) -> Outcome<()> {
            let width = TAPS * case.in_channels;
            let rows = padded_rows(case);
            let mut graph = ModelGraph::new(activation)?;
            let staged = graph.host(
                "staged",
                &[rows, width],
                seeded(&graph, (rows * width) as usize, 103),
            );
            let weight = graph.weight_taps(
                &format!("{}.weight", case.tensor),
                &[case.out_channels, width],
            );
            let output = project_rows(&mut graph.program, staged, weight, None)
                .map_err(context("linear"))?;
            let codec = DiffusionSettings::try_resolved()
                .map_err(context("diffusion settings"))?
                .vae_weights;
            with_tiles(tiles, |config| {
                let mut stage =
                    Decoder::new(checkpoint, graph, output, &[u64::from(rows)], codec, config)
                        .map_err(context("gemm stage"))?;
                measure(
                    &mut Arm {
                        label: "b_plain_gemm",
                        stage: &mut stage,
                        macs: conv_macs(case),
                    },
                    tiles,
                )
            })
        }

        fn form_d(activation: DType, codec: Codec, tiles: DenseMmaInputLayout) -> Outcome<()> {
            let mut graph = ModelGraph::new(activation)?;
            let input = graph.host(
                "control_activation",
                &[CONTROL_ROWS, CONTROL_DEPTH],
                seeded(&graph, (CONTROL_ROWS * CONTROL_DEPTH) as usize, 107),
            );
            let weight = graph.weight("control.weight", &[CONTROL_COLUMNS, CONTROL_DEPTH], None);
            let output =
                project_rows(&mut graph.program, input, weight, None).map_err(context("linear"))?;
            let bytes = noise_rows(&codec, (CONTROL_COLUMNS * CONTROL_DEPTH) as usize, 109)
                .map_err(context("control weight"))?
                .as_bytes()
                .to_vec();
            let weights = BoundWeights::new(&[]).with_packed_owned("control.weight", bytes, codec);
            with_tiles(tiles, |config| {
                // the arena cap scales with symbols[0], the plan's row count
                let mut stage = Decoder::with_weights(
                    graph,
                    output,
                    &[u64::from(CONTROL_ROWS)],
                    weights,
                    config,
                )
                .map_err(context("control stage"))?;
                measure(
                    &mut Arm {
                        label: "d_dit_shaped_linear",
                        stage: &mut stage,
                        macs: u64::from(CONTROL_ROWS)
                            * u64::from(CONTROL_DEPTH)
                            * u64::from(CONTROL_COLUMNS),
                    },
                    tiles,
                )
            })
        }

        pub fn run(checkpoint: &Checkpoint) -> Outcome<()> {
            let tiles = RuntimeConfig::from_env()
                .map_err(context("omega runtime variables"))?
                .dense_mma_input_layout;
            eprintln!("s0b_tiles layout={tiles:?}");
            let case = &CONV_CASES[1];
            let (activation, codec) = form_a(checkpoint, case, tiles)?;
            form_b(checkpoint, case, activation, tiles)?;
            form_d(activation, codec, tiles)
        }
    }

    fn print_table(rows: &[Row]) {
        eprintln!("flux2_klein_bench_table label,mean_ms,gmac_s,gb_s,iters");
        for row in rows {
            eprintln!(
                "flux2_klein_bench_row label={} {} iters={}",
                row.label,
                row.rates(),
                row.iterations,
            );
        }
    }

    fn wants(filters: &[String], group: &str, cases: &[&str]) -> bool {
        filters.is_empty()
            || filters.iter().any(|filter| {
                group.contains(filter.as_str())
                    || cases
                        .iter()
                        .any(|case| format!("{group}/{case}").contains(filter.as_str()))
            })
    }

    const VAE_CONV_CASES: [&str; 4] = [
        "window_same_128sq_32to512",
        "window_same_256sq_512to512",
        "window_same_512sq_256to256",
        "window_same_1024sq_128to128",
    ];
    const VAE_CASES: [&str; 4] = [
        "group_norm_128sq",
        "nearest_x2_128sq",
        "mid_attention_16384",
        "full_decode",
    ];
    const DIT_CASES: [&str; 4] = [
        "one_double_block",
        "one_single_block",
        "full_velocity_step",
        "attention_4608_score_mix",
    ];
    const TEXT_CASES: [&str; 2] = ["full_512_encode", "one_qwen3_layer"];

    pub fn main() -> ExitCode {
        match run() {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => {
                error!("flux2 klein bench stopped on a failed stage", %failure);
                ExitCode::FAILURE
            }
        }
    }

    fn run() -> Outcome<()> {
        let mut criterion = Criterion::default()
            .sample_size(10)
            .warm_up_time(Duration::from_millis(300))
            .measurement_time(Duration::from_secs(2))
            .configure_from_args();
        let filters = filters();
        let quick = std::env::args().any(|argument| argument == "--quick");
        let mut rows = Vec::new();
        // 1.2 GB staged activation and ~40 s of timed passes: only when named.
        let isolate_wanted = filters
            .iter()
            .any(|filter| "conv_isolate".contains(filter.as_str()));
        let vae_wanted = wants(&filters, "vae_conv", &VAE_CONV_CASES)
            || wants(&filters, "vae", &VAE_CASES)
            || isolate_wanted;
        if vae_wanted {
            let vae = Checkpoint::open(&vae_path()).map_err(context("open vae checkpoint"))?;
            #[cfg(feature = "instrument")]
            if isolate_wanted {
                isolate::run(&vae)?;
            }
            #[cfg(not(feature = "instrument"))]
            if isolate_wanted {
                return Err(String::from("conv_isolate needs --features instrument"));
            }
            if wants(&filters, "vae_conv", &VAE_CONV_CASES) {
                bench_vae_conv(&mut criterion, &mut rows, &vae)?;
            }
            if wants(&filters, "vae", &VAE_CASES) {
                bench_vae_norm_upsample(&mut criterion, &mut rows, &vae)?;
                bench_vae_attention(&mut criterion, &mut rows)?;
                // The full 1024^2 decode is ~42 s per iteration, so ten
                // criterion samples are seven minutes. `--quick` skips it
                // unless the run names it, which is how the full-stage cost
                // is measured on its own.
                let explicit_full = filters
                    .iter()
                    .any(|filter| "full_decode".contains(filter.as_str()));
                if !quick || explicit_full {
                    bench_vae_full(&mut criterion, &mut rows, &vae)?;
                }
            }
            drop(vae);
        }
        let dit_wanted = wants(&filters, "dit", &DIT_CASES);
        if dit_wanted {
            let dit = Checkpoint::open(&dit_path()).map_err(context("open dit checkpoint"))?;
            bench_dit_block(&mut criterion, &mut rows, &dit)?;
            bench_dit_attention(&mut criterion, &mut rows)?;
            bench_dit_full(&mut criterion, &mut rows, &dit)?;
            drop(dit);
        }
        if wants(&filters, "text", &TEXT_CASES) {
            bench_text(&mut criterion, &mut rows)?;
        }
        print_table(&rows);
        criterion.final_summary();
        Ok(())
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() -> ExitCode {
    bench::main()
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() -> ExitCode {
    eprintln!("the flux2-klein bench needs --features std,metal on macOS");
    ExitCode::from(64)
}
