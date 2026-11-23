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
//! iterations is a panic, never a silent pass.

#[cfg(all(feature = "metal", target_os = "macos"))]
mod bench {
    use std::cell::RefCell;
    use std::path::{Path, PathBuf};
    use std::rc::Rc;
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    use criterion::{BenchmarkGroup, Criterion, measurement::WallTime};
    use proxima_model_interop::diffusion::{
        Checkpoint, ConvTaps, DitConfig, DitInputs, Graph as ModelGraph, VaeConfig,
        build_conv3x3, build_dit, build_double_block, build_single_block, build_vae_decode,
        load_weights, timestep_sinusoid,
    };
    use proxima_model_interop::{
        ContextLength, GPU_LAYERS_ALL, LoadedModel, ServingConfig,
        architecture_from_hf_config, parse_hf_config,
    };
    use proxima_numeric::DType;
    use proxima_numeric::codec::Codec;
    use proxima_tensor::QuantizedBlock;
    use proxima_tensor::op::{NodeId, Op};
    use proxima_tensor::spec::{
        group_norm, nearest_upsample_x2, non_causal_attention_heads, unit_constant,
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

    /// Deterministic non-zero activation values: denormals and an all-zero
    /// operand are not what the real pipeline feeds these kernels.
    fn pseudo_random(count: usize, seed: u32) -> Vec<f32> {
        let mut state = seed | 1;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[derive(Default)]
    struct Stats {
        iterations: u64,
        samples: Vec<f64>,
    }

    struct Row {
        label: String,
        millis: f64,
        macs: f64,
        bytes: f64,
        iterations: u64,
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

    /// Runs one case, asserts it executed, and prints milliseconds, GMAC/s and
    /// GB/s from the measured iterations.
    fn case(
        group: &mut BenchmarkGroup<'_, WallTime>,
        group_label: &str,
        label: &str,
        macs: f64,
        bytes: f64,
        mut body: impl FnMut(),
    ) -> Row {
        let stats = Rc::new(RefCell::new(Stats::default()));
        let meter = Rc::clone(&stats);
        group.bench_function(label, |bencher| {
            bencher.iter(|| {
                let started = Instant::now();
                body();
                let elapsed = started.elapsed();
                let mut slot = meter.borrow_mut();
                slot.iterations += 1;
                slot.samples.push(elapsed.as_secs_f64());
            });
        });
        let stats = stats.borrow();
        let filters = filters();
        if selected(group_label, label, &filters) {
            assert!(
                stats.iterations > 0,
                "{group_label}/{label}: criterion executed zero iterations"
            );
        }
        // Median, not mean: the first iteration can include the pipeline's
        // one-time shader compile, and criterion's own warmup may not run a
        // whole slow iteration. The median is criterion's midpoint.
        let seconds = if stats.samples.is_empty() {
            0.0
        } else {
            let mut samples = stats.samples.clone();
            samples.sort_by(f64::total_cmp);
            let middle = samples.len() / 2;
            if samples.len() % 2 == 0 {
                (samples[middle - 1] + samples[middle]) / 2.0
            } else {
                samples[middle]
            }
        };
        let row = Row {
            label: label.to_string(),
            millis: seconds * 1e3,
            macs,
            bytes,
            iterations: stats.iterations,
        };
        if stats.iterations > 0 {
            eprintln!(
                "flux2_klein_bench case={group_label}/{} iters={} mean_ms={:.3} gmac_s={:.2} gb_s={:.2}",
                row.label,
                row.iterations,
                row.millis,
                gmac_per_second(row.macs, seconds),
                gb_per_second(row.bytes, seconds),
            );
        }
        row
    }

    fn gmac_per_second(macs: f64, seconds: f64) -> f64 {
        macs / seconds / 1e9
    }

    fn gb_per_second(bytes: f64, seconds: f64) -> f64 {
        bytes / seconds / 1e9
    }

    fn macs_of(program: &[Op], symbols: &[u64]) -> f64 {
        let shapes = proxima_tensor::infer(program, symbols).expect("program shapes");
        program
            .iter()
            .filter_map(|op| match op {
                Op::Reduce(reduce) => Some(
                    shapes
                        .of(reduce.operand)
                        .iter()
                        .map(|&extent| extent as f64)
                        .product::<f64>(),
                ),
                _ => None,
            })
            .sum()
    }

    fn plan_and_run(
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        output: NodeId,
    ) -> omega::backend::Plan {
        let (plan, _) = omega::backend::plan_named_with_config(
            &runtime(),
            program,
            symbols,
            named,
            &[output],
        )
        .expect("metal plan");
        plan
    }

    fn run_plan(plan: &mut omega::backend::Plan, named: &[(&str, QuantizedBlock<'_>)]) {
        let evaluated = omega::backend::execute_plan_named(plan, named).expect("metal execute");
        black_box(evaluated);
    }

    // ---------------------------------------------------------------- text --
    fn read_vocab(dir: &Path) -> Vocab {
        let tokenizer = std::fs::read(dir.join("tokenizer/tokenizer.json")).expect("tokenizer");
        let config_bytes = std::fs::read(dir.join("text_encoder/config.json")).expect("text config");
        let config: serde_json::Value = serde_json::from_slice(&config_bytes).expect("text json");
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
        .expect("tokenizer json")
    }

    fn padded_ids(vocab: &Vocab) -> Vec<u32> {
        let prompt = format!(
            "<|im_start|>user\na red fox sitting in fresh snow, morning light, photograph<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        let mut ids =
            proxima_tokenizer::encode_with_bos_eos(&prompt, vocab, false, false).expect("encode");
        ids.resize(MIN_LENGTH, 151_643);
        ids
    }

    struct TextModel {
        model: LoadedModel<'static>,
        _bytes: &'static [u8],
        _manifest: &'static proxima_safetensors::Manifest,
        roots: Vec<NodeId>,
        ids: Vec<u32>,
        serving: ServingConfig<'static>,
    }

    impl TextModel {
        fn load() -> Self {
            let dir = model_dir();
            let vocab = read_vocab(&dir);
            let ids = padded_ids(&vocab);
            let config_bytes = std::fs::read(dir.join("text_encoder/config.json")).expect("config");
            let hf_config = parse_hf_config(&config_bytes).expect("hf config");
            let architecture = architecture_from_hf_config(&hf_config).expect("architecture");
            let path = dir.join("comfy/qwen_3_4b.safetensors");
            // The text encoder is one shot per image; the bench process owns it
            // for the whole text group and leaks the mapping deliberately so
            // the model can hold a 'static borrow. The image path frees it.
            let bytes: &'static [u8] =
                Box::leak(std::fs::read(&path).expect("qwen checkpoint").into_boxed_slice());
            let manifest: &'static proxima_safetensors::Manifest =
                Box::leak(Box::new(proxima_safetensors::parse_complete(bytes).expect("manifest")));
            let data_start = {
                let mut prefix = [0u8; 8];
                prefix.copy_from_slice(&bytes[..8]);
                8 + u64::from_le_bytes(prefix) as usize
            };
            let model = LoadedModel::load_from_safetensors(
                manifest,
                bytes,
                data_start as u64,
                architecture,
                vocab.clone(),
            )
            .expect("load qwen");
            let roots = model.layer_residual_roots().to_vec();
            let mut serving = ServingConfig::default();
            serving.gpu_layers = GPU_LAYERS_ALL;
            serving.context_length = ContextLength::Within(MIN_LENGTH as u32);
            serving.model_path = Box::leak(
                dir.to_string_lossy().into_owned().into_boxed_str(),
            );
            Self {
                model,
                _bytes: bytes,
                _manifest: manifest,
                roots,
                ids,
                serving,
            }
        }
    }

    fn bench_text(c: &mut Criterion, rows: &mut Vec<Row>) {
        let text = TextModel::load();
        let taps: Vec<NodeId> = [8usize, 17, 26]
            .iter()
            .map(|&layer| text.roots[layer])
            .collect();
        let one_layer = [text.roots[0]];
        let tokens = TEXT_TOKENS as f64;

        let mut group = c.benchmark_group("text");
        group.sample_size(10);
        group.warm_up_time(Duration::from_millis(200));
        group.measurement_time(Duration::from_secs(2));
        // Qwen3 4B: hidden 2560, intermediate 9728, 32 query heads and 8 kv
        // heads of width 128 (SwiGLU MLP counted as three projections). The
        // 3-tap output stops after layer 27 (1-based), so the forward runs 27
        // of the model's 36 layers.
        let attention = 2560.0 * 4096.0 + 2.0 * 2560.0 * 1024.0 + 4096.0 * 2560.0;
        let mlp = 3.0 * 2560.0 * 9728.0;
        let full_macs = 27.0 * (attention + mlp) * tokens;
        let full_bytes =
            (3.0 * tokens * f64::from(JOINT_ATTENTION) + tokens * 2560.0 * 3.0) * 4.0;
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
                    .expect("full text encode");
                black_box(values.len());
            },
        ));
        let one_macs = (attention + mlp) * tokens;
        rows.push(case(
            &mut group,
            "text",
            "one_qwen3_layer",
            one_macs,
            (tokens * 2560.0 * 2.0) * 4.0,
            || {
                let values = text
                    .model
                    .forward_node_values_from_ids(&text.ids, &one_layer, &text.serving)
                    .expect("one text layer");
                black_box(values.len());
            },
        ));
        group.finish();
        drop(text);
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
        let cos = pseudo_random(positions, 7);
        let sin = pseudo_random(positions, 11);
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

    fn base_inputs(graph: &mut ModelGraph) -> (DitInputs, NodeId, NodeId) {
        let latent = graph.host(
            "latent",
            &[IMAGE_TOKENS, LATENT_CHANNELS],
            pseudo_random((IMAGE_TOKENS * LATENT_CHANNELS) as usize, 3),
        );
        let text = graph.host(
            "text",
            &[TEXT_TOKENS, JOINT_ATTENTION],
            pseudo_random((TEXT_TOKENS * JOINT_ATTENTION) as usize, 5),
        );
        let timestep = graph.host("timestep", &[256], timestep_sinusoid(0.5, 256));
        let (cos, sin) = rope_hosts(graph);
        (
            DitInputs {
                latent,
                text,
                timestep,
                rope_cos: cos,
                rope_sin: sin,
                image_tokens: IMAGE_TOKENS,
                text_tokens: TEXT_TOKENS,
            },
            latent,
            text,
        )
    }

    fn bench_dit_block(c: &mut Criterion, rows: &mut Vec<Row>, checkpoint: &Checkpoint) {
        // --- one double block ---
        {
            let mut graph = ModelGraph::new(DType::FLOAT32);
            let (inputs, _, _) = base_inputs(&mut graph);
            let image = graph.host(
                "block_image",
                &[IMAGE_TOKENS, HIDDEN],
                pseudo_random((IMAGE_TOKENS * HIDDEN) as usize, 61),
            );
            let text = graph.host(
                "block_text",
                &[TEXT_TOKENS, HIDDEN],
                pseudo_random((TEXT_TOKENS * HIDDEN) as usize, 67),
            );
            let mut image_mod = [NodeId(0); 6];
            let mut text_mod = [NodeId(0); 6];
            for (index, slot) in image_mod.iter_mut().enumerate() {
                *slot = graph.host(
                    &format!("image_mod{index}"),
                    &[HIDDEN],
                    pseudo_random(HIDDEN as usize, 13 + index as u32),
                );
            }
            for (index, slot) in text_mod.iter_mut().enumerate() {
                *slot = graph.host(
                    &format!("text_mod{index}"),
                    &[HIDDEN],
                    pseudo_random(HIDDEN as usize, 29 + index as u32),
                );
            }
            let (image, text_out) = build_double_block(
                &mut graph,
                &dit_config(1, 0),
                &inputs,
                "double_blocks.0",
                image,
                text,
                image_mod,
                text_mod,
            )
            .expect("double block");
            let weights =
                load_weights(checkpoint, &graph.weights, Codec::Float16, false).expect("dit weights");
            let integers = graph.int_buffers();
            let mut named = weights.blocks();
            named.extend(graph.host_blocks(&[], &integers));
            let symbols = [(MIN_LENGTH + IMAGE_TOKENS as usize) as u64];
            let macs = macs_of(&graph.program, &symbols);
            let bytes = dit_bytes(&graph.program, &symbols);
            let mut plan = plan_and_run(&graph.program, &symbols, &named, image);
            let mut group = c.benchmark_group("dit");
            group.sample_size(10);
            group.warm_up_time(Duration::from_millis(100));
            group.measurement_time(Duration::from_secs(1));
            rows.push(case(&mut group, "dit", "one_double_block", macs, bytes, || {
                run_plan(&mut plan, &named);
            }));
            group.finish();
            let _ = text_out;
        }
        // --- one single block ---
        {
            let mut graph = ModelGraph::new(DType::FLOAT32);
            let (inputs, _, _) = base_inputs(&mut graph);
            let stream = graph.host(
                "stream",
                &[(MIN_LENGTH + IMAGE_TOKENS as usize) as u32, HIDDEN],
                pseudo_random((MIN_LENGTH + IMAGE_TOKENS as usize) * HIDDEN as usize, 17),
            );
            let mut modulation = [NodeId(0); 3];
            for (index, slot) in modulation.iter_mut().enumerate() {
                *slot = graph.host(
                    &format!("single_mod{index}"),
                    &[HIDDEN],
                    pseudo_random(HIDDEN as usize, 41 + index as u32),
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
            .expect("single block");
            let weights =
                load_weights(checkpoint, &graph.weights, Codec::Float16, false).expect("dit weights");
            let integers = graph.int_buffers();
            let mut named = weights.blocks();
            named.extend(graph.host_blocks(&[], &integers));
            let symbols = [(MIN_LENGTH + IMAGE_TOKENS as usize) as u64];
            let macs = macs_of(&graph.program, &symbols);
            let bytes = dit_bytes(&graph.program, &symbols);
            let mut plan = plan_and_run(&graph.program, &symbols, &named, output);
            let mut group = c.benchmark_group("dit");
            group.sample_size(10);
            group.warm_up_time(Duration::from_millis(100));
            group.measurement_time(Duration::from_secs(1));
            rows.push(case(&mut group, "dit", "one_single_block", macs, bytes, || {
                run_plan(&mut plan, &named);
            }));
            group.finish();
        }
    }

    fn dit_bytes(program: &[Op], symbols: &[u64]) -> f64 {
        let shapes = proxima_tensor::infer(program, symbols).expect("program shapes");
        let count = |node: NodeId| -> f64 {
            shapes
                .of(node)
                .iter()
                .map(|&extent| extent as f64)
                .product::<f64>()
        };
        let mut bytes = 0.0;
        for (index, op) in program.iter().enumerate() {
            if let Op::Input { name, .. } = op {
                let element = if name
                    .as_deref()
                    .is_some_and(|name| name.contains(".weight"))
                {
                    2.0
                } else {
                    4.0
                };
                bytes += count(NodeId(index as u32)) * element;
            }
        }
        if let Some(last) = program.len().checked_sub(1) {
            bytes += count(NodeId(last as u32)) * 4.0;
        }
        bytes
    }

    fn bench_dit_full(c: &mut Criterion, rows: &mut Vec<Row>, checkpoint: &Checkpoint) {
        let mut graph = ModelGraph::new(DType::FLOAT32);
        let (inputs, _, _) = base_inputs(&mut graph);
        let output = build_dit(&mut graph, &dit_config(5, 20), &inputs).expect("full dit");
        let weights =
            load_weights(checkpoint, &graph.weights, Codec::Float16, false).expect("dit weights");
        let integers = graph.int_buffers();
        let mut named = weights.blocks();
        named.extend(graph.host_blocks(&[], &integers));
        let symbols = [(MIN_LENGTH + IMAGE_TOKENS as usize) as u64];
        let macs = macs_of(&graph.program, &symbols);
        let bytes = dit_bytes(&graph.program, &symbols);
        let mut plan = plan_and_run(&graph.program, &symbols, &named, output.velocity);
        let mut group = c.benchmark_group("dit");
        group.sample_size(10);
        group.warm_up_time(Duration::from_millis(200));
        group.measurement_time(Duration::from_secs(2));
        rows.push(case(&mut group, "dit", "full_velocity_step", macs, bytes, || {
            run_plan(&mut plan, &named);
        }));
        group.finish();
    }

    fn attention_program(
        graph: &mut ModelGraph,
        tokens: u32,
        heads: u32,
        head_dim: u32,
        seed: u32,
    ) -> NodeId {
        let half = head_dim / 2;
        let mut host = |name: &str, shape: &[u32], salt: u32| {
            let count: u32 = shape.iter().product();
            graph.host(
                name,
                shape,
                pseudo_random(count as usize, seed + salt),
            )
        };
        let q_even = host("q_even", &[tokens, heads, half], 1);
        let q_odd = host("q_odd", &[tokens, heads, half], 2);
        let k_even = host("k_even", &[tokens, heads, half], 3);
        let k_odd = host("k_odd", &[tokens, heads, half], 4);
        let v = host("v", &[tokens, heads, head_dim], 5);
        let activation = graph.activation();
        let scale = unit_constant(
            &mut graph.program,
            1.0 / (head_dim as f32).sqrt(),
            activation,
        );
        non_causal_attention_heads(
            &mut graph.program,
            q_even,
            q_odd,
            k_even,
            k_odd,
            v,
            scale,
            "u",
            "p",
            "d",
        )
        .expect("attention")
    }

    fn bench_dit_attention(c: &mut Criterion, rows: &mut Vec<Row>) {
        let tokens = MIN_LENGTH as u32 + IMAGE_TOKENS;
        let mut graph = ModelGraph::new(DType::FLOAT32);
        let output = attention_program(&mut graph, tokens, HEADS, HEAD_DIM, 53);
        let integers = graph.int_buffers();
        let named = graph.host_blocks(&[], &integers);
        let symbols = [tokens as u64];
        let macs = macs_of(&graph.program, &symbols);
        let bytes = (tokens as f64 * HEADS as f64 * HEAD_DIM as f64 * 2.0
            + tokens as f64 * HEADS as f64 * (HEAD_DIM as f64 / 2.0) * 4.0)
            * 4.0;
        let mut plan = plan_and_run(&graph.program, &symbols, &named, output);
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
            || run_plan(&mut plan, &named),
        ));
        group.finish();
    }

    // ----------------------------------------------------------------- vae --
    fn vae_config() -> VaeConfig {
        VaeConfig {
            channels: LATENT_CHANNELS / 4,
            block_channels: [128, 256, 512, 512],
            layers_per_block: 2,
            groups: 32,
            out_channels: 3,
            height: LATENT_SIDE * 2,
            width: LATENT_SIDE * 2,
            conv_taps: proxima_model_interop::diffusion::DiffusionSettings::resolved().conv_taps(),
        }
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
            label: "conv3x3_128sq_32to512",
            tensor: "decoder.conv_in",
            side: 128,
            in_channels: 32,
            out_channels: 512,
            groups: 32,
        },
        ConvCase {
            label: "conv3x3_256sq_512to512",
            tensor: "decoder.up.2.block.0.conv2",
            side: 256,
            in_channels: 512,
            out_channels: 512,
            groups: 32,
        },
        ConvCase {
            label: "conv3x3_512sq_256to256",
            tensor: "decoder.up.1.block.0.conv2",
            side: 512,
            in_channels: 256,
            out_channels: 256,
            groups: 32,
        },
        ConvCase {
            label: "conv3x3_1024sq_128to128",
            tensor: "decoder.up.0.block.1.conv2",
            side: 1024,
            in_channels: 128,
            out_channels: 128,
            groups: 32,
        },
    ];

    fn case_metrics(case: &ConvCase) -> (f64, f64) {
        let rows = f64::from(case.side) * f64::from(case.side);
        let macs = f64::from(case.out_channels)
            * rows
            * 9.0
            * f64::from(case.in_channels);
        let bytes = f64::from(case.out_channels) * f64::from(case.in_channels) * 9.0 * 2.0
            + rows * f64::from(case.in_channels) * 4.0
            + rows * f64::from(case.out_channels) * 4.0;
        (macs, bytes)
    }

    fn bench_vae_conv(c: &mut Criterion, rows: &mut Vec<Row>, checkpoint: &Checkpoint) {
        for case in &CONV_CASES {
            let rows_n = case.side * case.side;
            let mut graph = ModelGraph::new(DType::FLOAT32);
            let input = graph.host(
                "activation",
                &[rows_n, case.groups, case.in_channels / case.groups],
                pseudo_random(
                    (rows_n * case.in_channels) as usize,
                    101 + case.side,
                ),
            );
            let output = build_conv3x3(
                &mut graph,
                input,
                case.tensor,
                case.side,
                case.side,
                case.in_channels,
                case.out_channels,
                case.groups,
                case.groups,
                ConvTaps::Three,
            )
            .expect("conv");
            let weights =
                load_weights(checkpoint, &graph.weights, Codec::Float16, true).expect("conv weights");
            let integers = graph.int_buffers();
            let mut named = weights.blocks();
            named.extend(graph.host_blocks(&[], &integers));
            let symbols = [u64::from(rows_n)];
            let (macs, bytes) = case_metrics(case);
            let mut plan = plan_and_run(&graph.program, &symbols, &named, output);
            let mut group = c.benchmark_group("vae_conv");
            group.sample_size(10);
            group.warm_up_time(Duration::from_millis(100));
            group.measurement_time(Duration::from_secs(1));
            rows.push(case_for(
                &mut group,
                "vae_conv",
                case.label,
                macs,
                bytes,
                &mut plan,
                &named,
            ));
            group.finish();
        }
    }

    fn case_for(
        group: &mut BenchmarkGroup<'_, WallTime>,
        group_label: &str,
        label: &str,
        macs: f64,
        bytes: f64,
        plan: &mut omega::backend::Plan,
        named: &[(&str, QuantizedBlock<'_>)],
    ) -> Row {
        case(group, group_label, label, macs, bytes, || run_plan(plan, named))
    }

    fn bench_vae_norm_upsample(c: &mut Criterion, rows: &mut Vec<Row>, checkpoint: &Checkpoint) {
        // group_norm: the mid block's 512 channels at 128x128, 32 groups.
        {
            let mut graph = ModelGraph::new(DType::FLOAT32);
            let rows_n = 128u32 * 128u32;
            let channels = 512u32;
            let groups = 32u32;
            let x = graph.host(
                "activation",
                &[rows_n, groups, channels / groups],
                pseudo_random((rows_n * channels) as usize, 201),
            );
            let weight = graph.weight("decoder.mid.block_1.norm1.weight", &[groups, channels / groups], None);
            let bias = graph.weight("decoder.mid.block_1.norm1.bias", &[groups, channels / groups], None);
            let output = group_norm(&mut graph.program, x, weight, bias, rows_n * (channels / groups), 1e-6)
                .expect("group norm");
            let weights =
                load_weights(checkpoint, &graph.weights, Codec::Float16, true).expect("norm weights");
            let integers = graph.int_buffers();
            let mut named = weights.blocks();
            named.extend(graph.host_blocks(&[], &integers));
            let symbols = [u64::from(rows_n)];
            let bytes = (rows_n * channels * 2) as f64 * 4.0;
            let mut plan = plan_and_run(&graph.program, &symbols, &named, output);
            let mut group = c.benchmark_group("vae");
            group.sample_size(10);
            group.warm_up_time(Duration::from_millis(100));
            group.measurement_time(Duration::from_secs(1));
            rows.push(case_for(&mut group, "vae", "group_norm_128sq", 0.0, bytes, &mut plan, &named));
            group.finish();
        }
        // nearest_x2: the 128x128 -> 256x256 upsample that precedes a conv.
        {
            let mut graph = ModelGraph::new(DType::FLOAT32);
            let height = 128u32;
            let width = 128u32;
            let channels = 512u32;
            let groups = 32u32;
            let x = graph.host(
                "activation",
                &[height * width, groups, channels / groups],
                pseudo_random((height * width * channels) as usize, 211),
            );
            let index = graph.index_leaf("upsample.index", upsample_x2_index(height, width));
            let output = nearest_upsample_x2(&mut graph.program, x, index).expect("upsample");
            let integers = graph.int_buffers();
            let named = graph.host_blocks(&[], &integers);
            let symbols = [u64::from(height * width)];
            let bytes = (height * width * channels) as f64 * 4.0 * 2.0;
            let mut plan = plan_and_run(&graph.program, &symbols, &named, output);
            let mut group = c.benchmark_group("vae");
            group.sample_size(10);
            group.warm_up_time(Duration::from_millis(100));
            group.measurement_time(Duration::from_secs(1));
            rows.push(case_for(&mut group, "vae", "nearest_x2_128sq", 0.0, bytes, &mut plan, &named));
            group.finish();
        }
    }

    fn bench_vae_attention(c: &mut Criterion, rows: &mut Vec<Row>) {
        let tokens = 128u32 * 128u32;
        let mut graph = ModelGraph::new(DType::FLOAT32);
        // One head of width 512 split as 32 channel groups of 16; the same
        // shape the decoder's mid attention builds from its 1x1 projections.
        let output = attention_program(&mut graph, tokens, 32, 32, 71);
        let integers = graph.int_buffers();
        let named = graph.host_blocks(&[], &integers);
        let symbols = [u64::from(tokens)];
        let macs = macs_of(&graph.program, &symbols);
        let bytes = (tokens as f64 * 512.0 * 2.0) * 4.0;
        let mut plan = plan_and_run(&graph.program, &symbols, &named, output);
        let mut group = c.benchmark_group("vae");
        group.sample_size(10);
        group.warm_up_time(Duration::from_millis(100));
        group.measurement_time(Duration::from_secs(1));
        rows.push(case_for(
            &mut group,
            "vae",
            "mid_attention_16384",
            macs,
            bytes,
            &mut plan,
            &named,
        ));
        group.finish();
    }

    fn bench_vae_full(c: &mut Criterion, rows: &mut Vec<Row>, checkpoint: &Checkpoint) {
        let config = vae_config();
        let mut graph = ModelGraph::new(DType::FLOAT32);
        let tokens = LATENT_SIDE * 2 * LATENT_SIDE * 2;
        let input = graph.host(
            "latent",
            &[tokens, LATENT_CHANNELS / 4, 1],
            pseudo_random((tokens * LATENT_CHANNELS / 4) as usize, 301),
        );
        let output = build_vae_decode(&mut graph, &config, input).expect("vae decode");
        let weights =
            load_weights(checkpoint, &graph.weights, Codec::Float16, true).expect("vae weights");
        let integers = graph.int_buffers();
        let mut named = weights.blocks();
        named.extend(graph.host_blocks(&[], &integers));
        let symbols = [u64::from(tokens)];
        let macs = macs_of(&graph.program, &symbols);
        let bytes = 0.0;
        let mut plan = plan_and_run(&graph.program, &symbols, &named, output.image);
        let mut group = c.benchmark_group("vae");
        group.sample_size(10);
        group.warm_up_time(Duration::from_millis(200));
        group.measurement_time(Duration::from_secs(2));
        rows.push(case_for(
            &mut group,
            "vae",
            "full_decode",
            macs,
            bytes,
            &mut plan,
            &named,
        ));
        group.finish();
    }

    fn print_table(rows: &[Row]) {
        eprintln!(
            "flux2_klein_bench_table label,mean_ms,gmac_s,gb_s,iters"
        );
        for row in rows {
            let seconds = row.millis / 1e3;
            eprintln!(
                "flux2_klein_bench_row label={} mean_ms={:.3} gmac_s={:.2} gb_s={:.2} iters={}",
                row.label,
                row.millis,
                gmac_per_second(row.macs, seconds),
                gb_per_second(row.bytes, seconds),
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
        "conv3x3_128sq_32to512",
        "conv3x3_256sq_512to512",
        "conv3x3_512sq_256to256",
        "conv3x3_1024sq_128to128",
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

    pub fn main() {
        let mut criterion = Criterion::default()
            .sample_size(10)
            .warm_up_time(Duration::from_millis(300))
            .measurement_time(Duration::from_secs(2))
            .configure_from_args();
        let filters = filters();
        let quick = std::env::args().any(|argument| argument == "--quick");
        let mut rows = Vec::new();
        let vae_wanted = wants(&filters, "vae_conv", &VAE_CONV_CASES)
            || wants(&filters, "vae", &VAE_CASES);
        if vae_wanted {
            let vae = Checkpoint::open(&vae_path()).expect("open vae checkpoint");
            if wants(&filters, "vae_conv", &VAE_CONV_CASES) {
                bench_vae_conv(&mut criterion, &mut rows, &vae);
            }
            if wants(&filters, "vae", &VAE_CASES) {
                bench_vae_norm_upsample(&mut criterion, &mut rows, &vae);
                bench_vae_attention(&mut criterion, &mut rows);
                // The full 1024^2 decode is ~42 s per iteration, so ten
                // criterion samples are seven minutes. `--quick` skips it
                // unless the run names it, which is how the full-stage cost
                // is measured on its own.
                let explicit_full = filters.iter().any(|filter| "full_decode".contains(filter.as_str()));
                if !quick || explicit_full {
                    bench_vae_full(&mut criterion, &mut rows, &vae);
                }
            }
            drop(vae);
        }
        let dit_wanted = wants(&filters, "dit", &DIT_CASES);
        if dit_wanted {
            let dit = Checkpoint::open(&dit_path()).expect("open dit checkpoint");
            bench_dit_block(&mut criterion, &mut rows, &dit);
            bench_dit_attention(&mut criterion, &mut rows);
            bench_dit_full(&mut criterion, &mut rows, &dit);
            drop(dit);
        }
        if wants(&filters, "text", &TEXT_CASES) {
            bench_text(&mut criterion, &mut rows);
        }
        print_table(&rows);
        criterion.final_summary();
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    bench::main();
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("the flux2-klein bench needs --features std,metal on macOS");
    std::process::exit(64);
}
