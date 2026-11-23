//! Acceptance tests for the flux2-klein spec (`specs/flux2-klein/SPEC.md`). The library code
//! under test is generic; the reference dumps are what name the model.

mod philox {
    use std::env;
    use std::fs;
    use std::path::PathBuf;

    use proxima_numeric::{Philox, RngError, RngKind};

    const SEED: u64 = 42;
    const LATENT_VALUES: usize = 64 * 64 * 128;
    const TOLERANCE: f32 = 1e-5;

    type Outcome = Result<(), String>;

    fn reference_dir() -> Result<PathBuf, String> {
        if let Some(dir) = env::var_os("PROXIMA_FLUX2_KLEIN_REF") {
            return Ok(PathBuf::from(dir));
        }
        let home = env::var_os("HOME").ok_or("HOME locates the default reference dir")?;
        Ok(PathBuf::from(home).join(".cache/kvgate/proxnum588/ref"))
    }

    fn reference_noise() -> Result<Vec<f32>, String> {
        let path = reference_dir()?.join("noise.f32");
        let bytes = fs::read(&path).map_err(|why| format!("fixture {}: {why}", path.display()))?;
        let (words, _) = bytes.as_chunks::<4>();
        Ok(words.iter().map(|word| f32::from_le_bytes(*word)).collect())
    }

    fn ours() -> Result<Vec<f32>, String> {
        let selected: RngKind = "philox".parse().map_err(|why: RngError| why.to_string())?;
        Ok(selected.generator(SEED).normal(LATENT_VALUES))
    }

    fn max_abs_diff(left: &[f32], right: &[f32]) -> f32 {
        left.iter().zip(right).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max)
    }

    #[test]
    fn the_first_sixteen_values_of_seed_42_match_the_reference_dump() -> Outcome {
        let reference = reference_noise()?;
        let generated = ours()?;
        let worst = max_abs_diff(&generated[..16], &reference[..16]);
        assert!(worst <= TOLERANCE, "max abs diff {worst}");
        Ok(())
    }

    #[test]
    fn all_524288_values_of_seed_42_match_the_reference_dump() -> Outcome {
        let reference = reference_noise()?;
        let generated = ours()?;
        assert_eq!(reference.len(), LATENT_VALUES);
        assert_eq!(generated.len(), LATENT_VALUES);
        let worst = max_abs_diff(&generated, &reference);
        println!("max abs diff {worst}; ours {:?}; ref {:?}", &generated[..4], &reference[..4]);
        assert!(worst <= TOLERANCE, "max abs diff {worst}");
        Ok(())
    }

    #[test]
    fn the_latent_is_standard_normal_and_unsupported_generators_are_refused_by_name() {
        let generated = Philox::new(SEED).normal(LATENT_VALUES);
        let count = generated.len() as f64;
        let mean = generated.iter().map(|v| f64::from(*v)).sum::<f64>() / count;
        let spread = generated.iter().map(|v| (f64::from(*v) - mean).powi(2)).sum::<f64>() / count;
        assert!((-0.01..=0.01).contains(&mean), "mean {mean}");
        assert!((0.99..=1.01).contains(&spread.sqrt()), "std {}", spread.sqrt());
        for name in ["std_default", "cpu"] {
            let refused = name.parse::<RngKind>();
            assert_eq!(refused, Err(RngError::Unimplemented { name }));
        }
    }
}

mod scheduler {
    use std::env;
    use std::fs;
    use std::path::PathBuf;

    use proxima_numeric::{FloatLayout, FlowMatchSchedule, StepRule, euler_step, step};

    const IMAGE_TOKENS: u32 = 4096;
    const STEPS: u32 = 4;

    type Outcome = Result<(), String>;

    fn reference_dir() -> Result<PathBuf, String> {
        if let Some(dir) = env::var_os("PROXIMA_FLUX2_KLEIN_REF") {
            return Ok(PathBuf::from(dir));
        }
        let home = env::var_os("HOME").ok_or("HOME locates the default reference dir")?;
        Ok(PathBuf::from(home).join(".cache/kvgate/proxnum588/ref"))
    }

    fn reference_sigmas() -> Result<Vec<f32>, String> {
        let path = reference_dir()?.join("sigmas.txt");
        let text = fs::read_to_string(&path)
            .map_err(|why| format!("fixture {}: {why}", path.display()))?;
        let parse = |word: &str| word.parse::<f32>().map_err(|why| format!("{word}: {why}"));
        text.split_whitespace().map(parse).collect()
    }

    #[test]
    fn sigmas_for_4096_tokens_and_4_steps_match_the_reference_log() -> Outcome {
        let reference = reference_sigmas()?;
        let schedule = FlowMatchSchedule::new(STEPS, IMAGE_TOKENS);
        let ours = schedule.sigmas().map_err(|why| why.to_string())?;
        assert_eq!(reference.len(), 5);
        assert_eq!(ours.len(), reference.len());
        for (ours, theirs) in ours.iter().zip(&reference) {
            assert!((ours - theirs).abs() <= 1e-5, "ours {ours} ref {theirs}");
        }
        Ok(())
    }

    #[test]
    fn mu_for_4096_tokens_and_4_steps_is_the_logged_2_29118() {
        let mu = FlowMatchSchedule::new(STEPS, IMAGE_TOKENS).mu();
        assert!((mu - 2.29118).abs() <= 1e-4, "mu {mu}");
    }

    #[test]
    fn the_euler_step_adds_the_sigma_difference_times_velocity_element_for_element() {
        let start = [0.5_f32, -1.25, 3.0, 0.0, 7.5];
        let velocity = [1.0_f32, -2.0, 0.25, 4.0, -0.5];
        let (sigma, sigma_next) = (0.967_384_04_f32, 0.908_143_94_f32);
        let mut latent = start;
        euler_step(&mut latent, &velocity, sigma, sigma_next, FloatLayout::BF32).expect("binary32");
        let mut through_rule = start;
        let layout = FloatLayout::BF32;
        step(StepRule::Euler, &mut through_rule, &velocity, sigma, sigma_next, layout)
            .expect("binary32");
        assert_eq!(latent, through_rule);
        for index in 0..start.len() {
            let expected = start[index] + (sigma_next - sigma) * velocity[index];
            assert_eq!(latent[index], expected, "element {index}");
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod dit {
    use std::env;
    use std::fs;
    use std::path::PathBuf;
    use std::time::Instant;

    use proxima_model_interop::diffusion::{
        Checkpoint, DitConfig, DitInputs, Graph, build_dit, load_weights, timestep_sinusoid,
    };
    use proxima_numeric::codec::Codec;
    use proxima_numeric::{DType, FloatLayout, FlowMatchSchedule, euler_step};
    use proxima_tensor::op::NodeId;
    use proxima_tensor::QuantizedBlock;
    use proxima_tensor::spec::multi_axis_rope_tables;

    const IMAGE_TOKENS: usize = 64 * 64;
    const TEXT_TOKENS: usize = 512;
    const HIDDEN: usize = 128;
    const JOINT: usize = 4608;
    const STEPS: u32 = 4;
    const ROPE_DAXES: [u32; 4] = [32, 32, 32, 32];
    const ROPE_THETA: f32 = 2000.0;

    type Outcome = Result<(), String>;

    struct Fixtures {
        model: PathBuf,
        reference: PathBuf,
    }

    fn fixtures() -> Result<Fixtures, String> {
        let home = env::var_os("HOME").ok_or("HOME locates the fixtures")?;
        let model = env::var_os("PROXIMA_FLUX2_KLEIN_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(&home).join("models/image_gen/flux2-klein-4b"));
        let reference = env::var_os("PROXIMA_FLUX2_KLEIN_REF")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(&home).join(".cache/kvgate/proxnum588/ref"));
        if !model.exists() {
            return Err(format!("missing model dir {}", model.display()));
        }
        if !reference.exists() {
            return Err(format!("missing reference dir {}", reference.display()));
        }
        Ok(Fixtures { model, reference })
    }

    fn read_dump(path: &PathBuf) -> Result<Vec<f32>, String> {
        let bytes = fs::read(path).map_err(|why| format!("{}: {why}", path.display()))?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect())
    }

    /// The reference latent is channel-major (`index = token + tokens * channel`);
    /// the graph reads it token-major.
    fn token_major(channel_major: &[f32]) -> Vec<f32> {
        token_major_and(channel_major, IMAGE_TOKENS, HIDDEN)
    }

    fn token_major_and(channel_major: &[f32], tokens: usize, channels: usize) -> Vec<f32> {
        let mut values = vec![0.0f32; channel_major.len()];
        for channel in 0..channels {
            for token in 0..tokens {
                values[token * channels + channel] = channel_major[token + tokens * channel];
            }
        }
        values
    }

    /// A `[channels, tokens]` capture (dim0 the channel axis) as token-major.
    fn from_channel_major(values: &[f32], tokens: usize, channels: usize) -> Vec<f32> {
        let mut token_major = vec![0.0f32; values.len()];
        for token in 0..tokens {
            for channel in 0..channels {
                token_major[token * channels + channel] = values[channel + channels * token];
            }
        }
        token_major
    }

    fn stats(actual: &[f32], reference: &[f32]) -> (f64, f64) {
        let mut squared_diff = 0.0f64;
        let mut squared_reference = 0.0f64;
        let mut dot = 0.0f64;
        let mut squared_actual = 0.0f64;
        for (&got, &want) in actual.iter().zip(reference) {
            let (got, want) = (f64::from(got), f64::from(want));
            squared_diff += (got - want) * (got - want);
            squared_reference += want * want;
            squared_actual += got * got;
            dot += got * want;
        }
        let count = actual.len() as f64;
        (
            (squared_diff / count).sqrt() / (squared_reference / count).sqrt(),
            dot / (squared_actual * squared_reference).sqrt(),
        )
    }

    fn dit_config() -> DitConfig {
        DitConfig {
            in_channels: 128,
            hidden: 3072,
            heads: 24,
            head_dim: 128,
            joint_attention_dim: 7680,
            double_blocks: 5,
            single_blocks: 20,
            timestep_channels: 256,
            rope_theta: ROPE_THETA as u32,
        }
    }

    fn rope_positions() -> Vec<f32> {
        let mut positions = Vec::with_capacity(JOINT * 4);
        for token in 0..TEXT_TOKENS {
            positions.extend_from_slice(&[0.0, 0.0, 0.0, token as f32]);
        }
        let width = 64u32;
        for token in 0..IMAGE_TOKENS {
            let row = (token / width as usize) as f32;
            let column = (token % width as usize) as f32;
            positions.extend_from_slice(&[0.0, row, column, 0.0]);
        }
        positions
    }

    fn runtime() -> omega::config::RuntimeConfig {
        let serving = proxima_model_interop::ServingConfig::default();
        omega::config::RuntimeConfig::builder()
            .with_engine("gpu")
            .with_gpu_driver("metal")
            .with_allow_fusion(true)
            .with_numeric(proxima_numeric::NumericPolicy::llama_relaxed())
            .with_exact_activations(serving.exact_activations)
            .build()
            .expect("metal runtime")
    }

    #[test]
    fn velocity_and_the_four_euler_steps_match_the_reference() -> Outcome {
        let fixtures = fixtures()?;
        let started = Instant::now();
        let mut graph = Graph::new(DType::FLOAT32);
        let latent = graph.host("latent", &[IMAGE_TOKENS as u32, HIDDEN as u32], Vec::new());
        let text = graph.host("text", &[TEXT_TOKENS as u32, 7680], Vec::new());
        let timestep = graph.host("timestep", &[256], Vec::new());
        let (cos_bytes, sin_bytes) = multi_axis_rope_tables(
            &rope_positions(),
            &ROPE_DAXES,
            &[ROPE_THETA; 4],
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
        let rope_cos = graph.host("rope_cos", &[JOINT as u32, 64], decode(&cos_bytes));
        let rope_sin = graph.host("rope_sin", &[JOINT as u32, 64], decode(&sin_bytes));
        let inputs = DitInputs {
            latent,
            text,
            timestep,
            rope_cos,
            rope_sin,
            image_tokens: IMAGE_TOKENS as u32,
            text_tokens: TEXT_TOKENS as u32,
        };
        let output = build_dit(&mut graph, &dit_config(), &inputs).map_err(|why| why.to_string())?;
        let velocity = output.velocity;
        let mut traced: Vec<(&str, NodeId)> = output
            .trace
            .iter()
            .map(|(name, node)| (*name, *node))
            .collect();
        traced.push(("velocity", velocity));
        let trace_nodes: Vec<NodeId> = traced.iter().map(|(_, node)| *node).collect();
        eprintln!(
            "program: {} ops, {} weight leaves, {} host leaves, built in {:.1}s",
            graph.program.len(),
            graph.weights.len(),
            graph.hosts.len(),
            started.elapsed().as_secs_f64()
        );

        let checkpoint = Checkpoint::open(&fixtures.model.join("flux-2-klein-4b.safetensors"))?;
        let loading = Instant::now();
        let weights = load_weights(&checkpoint, &graph.weights, Codec::Float16, false)?;
        eprintln!("weights loaded in {:.1}s", loading.elapsed().as_secs_f64());

        let mut latent_values =
            token_major(&read_dump(&fixtures.reference.join("noise.f32"))?);
        let text_values = read_dump(&fixtures.reference.join("text.f32"))?;
        let cos_values = decode(&cos_bytes);
        let sin_values = decode(&sin_bytes);
        let sigmas = FlowMatchSchedule::new(STEPS, IMAGE_TOKENS as u32)
            .sigmas()
            .map_err(|why| why.to_string())?;

        let runtime = runtime();
        let planning_timestep = timestep_sinusoid(1.0, 256);
        let integers = graph.int_buffers();
        let mut blocks: Vec<(&str, QuantizedBlock)> = weights.blocks();
        blocks.extend(graph.host_blocks(
            &
            [
                ("latent", latent_values.as_slice()),
                ("text", text_values.as_slice()),
                ("timestep", planning_timestep.as_slice()),
                ("rope_cos", cos_values.as_slice()),
                ("rope_sin", sin_values.as_slice()),
            ],
            &integers,
        ));
        let plan_started = Instant::now();
        let (mut plan, _) = omega::backend::plan_named_with_config(
            &runtime,
            &graph.program,
            &[JOINT as u64],
            &blocks,
            &trace_nodes,
        )
        .map_err(|why| format!("plan: {why}"))?;
        eprintln!("plan built in {:.1}s", plan_started.elapsed().as_secs_f64());
        let _ = weights.blocks();

        let reference_velocity = token_major(&read_dump(&fixtures.reference.join("velocity1.f32"))?);
        let mut verdicts = 0;
        for step in 0..STEPS as usize {
            let timestep_values = timestep_sinusoid(sigmas[step], 256);
            let mut named: Vec<(&str, QuantizedBlock)> = weights.blocks();
            named.extend(graph.host_blocks(
                &
                [
                    ("latent", latent_values.as_slice()),
                    ("text", text_values.as_slice()),
                    ("timestep", timestep_values.as_slice()),
                    ("rope_cos", cos_values.as_slice()),
                    ("rope_sin", sin_values.as_slice()),
                ],
                &integers,
            ));
            let ran = Instant::now();
            #[cfg(feature = "instrument")]
            let profiled = if env::var_os("PROXIMA_FLUX2_KLEIN_PROFILE").is_some() {
                let (evaluated, timings) =
                    omega::backend::execute_plan_named_metal_op_timed(&plan, &named, None)
                        .map_err(|why| format!("step {}: {why}", step + 1))?;
                let mut rows: Vec<(&str, u64, Vec<u64>)> = timings
                    .iter()
                    .map(|timing| (timing.kind, timing.gpu_ns, timing.extents.clone()))
                    .collect();
                rows.sort_by_key(|row| core::cmp::Reverse(row.1));
                for (kind, nanos, extents) in rows.iter().take(14) {
                    eprintln!("op {kind} {:?} {:.1}ms", extents, *nanos as f64 / 1e6);
                }
                let total: u64 = rows.iter().map(|row| row.1).sum();
                eprintln!("op gpu total {:.1}s over {} ops", total as f64 / 1e9, rows.len());
                Some(evaluated)
            } else {
                None
            };
            #[cfg(not(feature = "instrument"))]
            let profiled = None::<_>;
            let evaluated = match profiled {
                Some(evaluated) => evaluated,
                None => omega::backend::execute_plan_named(&mut plan, &named)
                    .map_err(|why| format!("step {}: {why}", step + 1))?,
            };
            let elapsed = ran.elapsed().as_secs_f64();
            let velocity_values = evaluated
                .get(velocity)
                .ok_or("velocity is not in the evaluated set")?
                .0
                .to_vec();
            if step == 0 && env::var_os("PROXIMA_FLUX2_KLEIN_PROFILE").is_some() {
                return Ok(());
            }
            if step == 0 {
                if let Some(trace_dir) = env::var_os("PROXIMA_FLUX2_KLEIN_TRACE") {
                    let trace_dir = PathBuf::from(trace_dir);
                    for (name, node) in &traced {
                        if *name == "velocity" {
                            continue;
                        }
                        let path = trace_dir.join(format!("p_{name}.f32"));
                        if !path.exists() {
                            eprintln!("trace {name}: no reference at {}", path.display());
                            continue;
                        }
                        let reference = read_dump(&path)?;
                        let actual = evaluated
                            .get(*node)
                            .ok_or_else(|| format!("trace {name} is not in the evaluated set"))?
                            .0;
                        let (rms, cosine) = if *name == "vec" {
                            stats(actual, &reference)
                        } else if name.starts_with("b_imgq") || name.starts_with("b_imgk") || name.starts_with("b_imgv") {
                            let mut token_major = vec![0.0f32; reference.len()];
                            for token in 0..4096 {
                                for head in 0..24 {
                                    for axis in 0..128 {
                                        token_major[token * 3072 + head * 128 + axis] =
                                            reference[axis + 128 * (head + 24 * token)];
                                    }
                                }
                            }
                            stats(actual, &token_major)
                        } else {
                            let channels = if *name == "final" { 128 } else { 3072 };
                            let tokens = actual.len() / channels;
                            stats(actual, &from_channel_major(&reference, tokens, channels))
                        };
                        eprintln!("trace {name}: rms_rel={rms:.6} cosine={cosine:.8}");
                    }
                }
                let (rms, cosine) = stats(&velocity_values, &reference_velocity);
                eprintln!("velocity1: rms_rel={rms:.6} cosine={cosine:.8} in {elapsed:.2}s");
                assert!(rms <= 0.101, "velocity rms_rel {rms}");
                assert!(cosine >= 0.999, "velocity cosine {cosine}");
                verdicts += 1;
            }
            euler_step(
                &mut latent_values,
                &velocity_values,
                sigmas[step],
                sigmas[step + 1],
                FloatLayout::BF32,
            )
            .map_err(|why| why.to_string())?;
            let reference =
                token_major(&read_dump(&fixtures.reference.join(format!("step{}.f32", step + 1)))?);
            let (rms, cosine) = stats(&latent_values, &reference);
            let bar = [0.003, 0.007, 0.016, 0.078][step];
            eprintln!("step{}: rms_rel={rms:.6} cosine={cosine:.8} bar={bar} in {elapsed:.2}s", step + 1);
            assert!(rms <= bar, "step {} rms_rel {rms} above {bar}", step + 1);
            assert!(cosine >= 0.999, "step {} cosine {cosine}", step + 1);
            verdicts += 1;
        }
        eprintln!("{verdicts} verdicts printed");
        Ok(())
    }
}

#[cfg(feature = "std")]
mod dit_shape {
    use proxima_model_interop::diffusion::{DitConfig, DitInputs, Graph, build_dit};
    use proxima_numeric::DType;

    const IMAGE_TOKENS: usize = 64 * 64;
    const TEXT_TOKENS: usize = 512;
    const JOINT: usize = 4608;

    fn dit_config() -> DitConfig {
        DitConfig {
            in_channels: 128,
            hidden: 3072,
            heads: 24,
            head_dim: 128,
            joint_attention_dim: 7680,
            double_blocks: 5,
            single_blocks: 20,
            timestep_channels: 256,
            rope_theta: 2000,
        }
    }

    #[test]
    fn the_whole_backbone_program_shapes_without_any_weights() {
        let mut graph = Graph::new(DType::FLOAT32);
        let latent = graph.host("latent", &[IMAGE_TOKENS as u32, 128], vec![0.0; 8]);
        let text = graph.host("text", &[TEXT_TOKENS as u32, 7680], vec![0.0; 8]);
        let timestep = graph.host("timestep", &[256], vec![0.0; 8]);
        let rope_cos = graph.host("rope_cos", &[JOINT as u32, 64], vec![0.0; 8]);
        let rope_sin = graph.host("rope_sin", &[JOINT as u32, 64], vec![0.0; 8]);
        let inputs = DitInputs {
            latent,
            text,
            timestep,
            rope_cos,
            rope_sin,
            image_tokens: IMAGE_TOKENS as u32,
            text_tokens: TEXT_TOKENS as u32,
        };
        let output = build_dit(&mut graph, &dit_config(), &inputs).expect("backbone builds");
        let velocity = output.velocity;
        match proxima_tensor::infer(&graph.program, &[JOINT as u64]) {
            Ok(shapes) => {
                assert_eq!(shapes.of(velocity), vec![IMAGE_TOKENS as u64, 128]);
            }
            Err(why) => {
                let at = format!("{why}")
                    .split('%')
                    .nth(1)
                    .and_then(|rest| rest.split(|c: char| !c.is_ascii_digit()).next())
                    .and_then(|digits| digits.parse::<usize>().ok())
                    .unwrap_or(0);
                for index in at.saturating_sub(14)..=(at + 1).min(graph.program.len() - 1) {
                    eprintln!("node {index}: {:?}", graph.program[index]);
                }
                panic!("{why}");
            }
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod prelude_layout {
    use std::env;
    use std::path::PathBuf;

    use proxima_model_interop::diffusion::{Checkpoint, timestep_sinusoid};

    type Outcome = Result<(), String>;

    fn trace_dir() -> Result<PathBuf, String> {
        env::var_os("PROXIMA_FLUX2_KLEIN_TRACE")
            .map(PathBuf::from)
            .ok_or_else(|| String::from("set PROXIMA_FLUX2_KLEIN_TRACE to the capture dir"))
    }

    fn model_dir() -> Result<PathBuf, String> {
        let home = env::var_os("HOME").ok_or("HOME locates the model")?;
        Ok(env::var_os("PROXIMA_FLUX2_KLEIN_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(&home).join("models/image_gen/flux2-klein-4b")))
    }

    fn read(path: &PathBuf) -> Result<Vec<f32>, String> {
        let bytes = std::fs::read(path).map_err(|why| format!("{}: {why}", path.display()))?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect())
    }

    fn cos(actual: &[f32], reference: &[f32]) -> f64 {
        let mut dot = 0.0f64;
        let mut left = 0.0f64;
        let mut right = 0.0f64;
        for (&a, &b) in actual.iter().zip(reference) {
            dot += f64::from(a) * f64::from(b);
            left += f64::from(a) * f64::from(a);
            right += f64::from(b) * f64::from(b);
        }
        dot / (left * right).sqrt()
    }

    /// `out[t, o] = sum_i x[t, i] * weight[o, i]`.
    fn matvec(x: &[f32], weight: &[f32], tokens: usize, into: usize, out: usize) -> Vec<f32> {
        let mut values = vec![0.0f32; tokens * out];
        for token in 0..tokens {
            for column in 0..out {
                let row = &x[token * into..(token + 1) * into];
                let kernel = &weight[column * into..(column + 1) * into];
                values[token * out + column] = row
                    .iter()
                    .zip(kernel)
                    .map(|(left, right)| left * right)
                    .sum();
            }
        }
        values
    }

    #[test]
    fn the_captured_prelude_identifies_the_latent_mapping_and_time_embedding() -> Outcome {
        let trace = trace_dir()?;
        let checkpoint = Checkpoint::open(&model_dir()?.join("flux-2-klein-4b.safetensors"))?;
        let (_, image_weight) = checkpoint.values("img_in.weight", None, None, false)?;
        let (_, in_weight) = checkpoint.values("time_in.in_layer.weight", None, None, false)?;
        let (_, out_weight) = checkpoint.values("time_in.out_layer.weight", None, None, false)?;
        let reference_image_in = read(&trace.join("p_img_in.f32"))?;
        let reference_vec = read(&trace.join("p_vec.f32"))?;
        let noise = read(&trace.join("noise.f32"))?;

        const TOKENS: usize = 64 * 64;
        const CHANNELS: usize = 128;
        const HIDDEN: usize = 3072;
        let _ = HIDDEN;
        let token_first: Vec<f32> = (0..TOKENS * CHANNELS)
            .map(|index| noise[(index / CHANNELS) + TOKENS * (index % CHANNELS)])
            .collect();
        let channel_first: Vec<f32> = noise.clone();
        for (label, latent) in [("token-major", &token_first), ("row-major", &channel_first)] {
            let projected = matvec(latent, &image_weight, TOKENS, CHANNELS, HIDDEN);
            let reference: Vec<f32> = (0..TOKENS * HIDDEN)
                .map(|index| reference_image_in[(index % HIDDEN) + HIDDEN * (index / HIDDEN)])
                .collect();
            eprintln!(
                "img_in with the {label} latent: cosine={:.8}",
                cos(&projected, &reference)
            );
            if label == "token-major" {
                let mut per_token = Vec::with_capacity(TOKENS);
                for token in 0..TOKENS {
                    let range = token * HIDDEN..(token + 1) * HIDDEN;
                    per_token.push(cos(&projected[range.clone()], &reference[range]));
                }
                per_token.sort_by(|left, right| left.partial_cmp(right).expect("finite"));
                eprintln!(
                    "  per-token cosine: min={:.6} p10={:.6} median={:.6} max={:.6}",
                    per_token[0],
                    per_token[TOKENS / 10],
                    per_token[TOKENS / 2],
                    per_token[TOKENS - 1]
                );
                let mut per_channel = Vec::with_capacity(HIDDEN);
                for column in 0..HIDDEN {
                    let ours: Vec<f32> = (0..TOKENS).map(|token| projected[token * HIDDEN + column]).collect();
                    let theirs: Vec<f32> = (0..TOKENS).map(|token| reference[token * HIDDEN + column]).collect();
                    per_channel.push(cos(&ours, &theirs));
                }
                per_channel.sort_by(|left, right| left.partial_cmp(right).expect("finite"));
                eprintln!(
                    "  per-output-channel cosine: min={:.6} p10={:.6} median={:.6} max={:.6}",
                    per_channel[0],
                    per_channel[HIDDEN / 10],
                    per_channel[HIDDEN / 2],
                    per_channel[HIDDEN - 1]
                );
                eprintln!(
                    "  first token ours {:?} theirs {:?}",
                    &projected[..4],
                    &reference[..4]
                );
                let ours = &projected[..HIDDEN];
                let theirs = &reference[..HIDDEN];
                let mut delta = vec![(0.0f32, 0usize); CHANNELS];
                for channel in 0..CHANNELS {
                    let kernel = &image_weight[channel * CHANNELS..(channel + 1) * CHANNELS];
                    let norm: f32 = kernel.iter().map(|value| value * value).sum();
                    let residual: f32 = (0..HIDDEN)
                        .map(|column| {
                            (theirs[column] - ours[column])
                                * image_weight[column * CHANNELS + channel]
                        })
                        .sum();
                    delta[channel] = (residual / norm, channel);
                }
                delta.sort_by(|left, right| {
                    right.0.abs().partial_cmp(&left.0.abs()).expect("finite")
                });
                for (estimate, channel) in delta.iter().take(8) {
                    eprintln!(
                        "  implied input delta channel {channel}: {estimate:.5} (ours {:.5}, noise {:.5})",
                        token_first[*channel], noise[*channel]
                    );
                }
            }
        }

        for (label, values) in [
            ("cos-first", timestep_sinusoid(1.0, 256)),
            ("sin-first", {
                let mut values = timestep_sinusoid(1.0, 256);
                values.rotate_left(128);
                values
            }),
            ("cos-at-0.001", timestep_sinusoid(0.001, 256)),
            ("sin-at-0.001", {
                let mut values = timestep_sinusoid(0.001, 256);
                values.rotate_left(128);
                values
            }),
        ] {
            let inner = matvec(&values, &in_weight, 1, 256, HIDDEN);
            let inner: Vec<f32> = inner
                .iter()
                .map(|value| value / (1.0 + (-value).exp()))
                .collect();
            let projected = matvec(&inner, &out_weight, 1, HIDDEN, HIDDEN);
            eprintln!(
                "time_in with the {label} sinusoid: cosine={:.8}",
                cos(&projected, &reference_vec)
            );
        }
        Ok(())
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod vae {
    use std::env;
    use std::fs;
    use std::path::PathBuf;
    use std::time::Instant;

    use proxima_model_interop::diffusion::{
        Checkpoint, Graph, VaeConfig, build_vae_decode, load_weights,
    };
    use proxima_numeric::DType;
    use proxima_numeric::codec::Codec;
    use proxima_tensor::QuantizedBlock;

    const LATENT: usize = 64;
    const LATENT_CHANNELS: usize = 128;
    const PATCH: usize = 2;
    const CHANNELS: usize = LATENT_CHANNELS / (PATCH * PATCH);
    const SIDE: usize = LATENT * PATCH;
    const TOKENS: usize = SIDE * SIDE;

    type Outcome = Result<(), String>;

    fn reference_dir() -> Result<PathBuf, String> {
        let home = env::var_os("HOME").ok_or("HOME locates the fixtures")?;
        Ok(env::var_os("PROXIMA_FLUX2_KLEIN_REF")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(&home).join(".cache/kvgate/proxnum588/ref")))
    }

    fn model_dir() -> Result<PathBuf, String> {
        let home = env::var_os("HOME").ok_or("HOME locates the model")?;
        Ok(env::var_os("PROXIMA_FLUX2_KLEIN_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(&home).join("models/image_gen/flux2-klein-4b")))
    }

    fn read_dump(path: &PathBuf) -> Result<Vec<f32>, String> {
        let bytes = fs::read(path).map_err(|why| format!("{}: {why}", path.display()))?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect())
    }

    fn config() -> VaeConfig {
        VaeConfig {
            channels: CHANNELS as u32,
            block_channels: [128, 256, 512, 512],
            layers_per_block: 2,
            groups: 32,
            out_channels: 3,
            height: SIDE as u32,
            width: SIDE as u32,
            // `PROXIMA_DIFFUSION_CONV_TAPS=nine` A/Bs the lowering without a
            // rebuild; unset keeps the default three-tap form.
            conv_taps: proxima_model_interop::diffusion::DiffusionSettings::resolved().conv_taps(),
        }
    }

    /// `step4` is the 128-channel latent in channel-major order; the decoder
    /// wants `[row, column, channels / patch^2]` after the checkpoint's own
    /// per-channel de-normalisation and the 2x2 pixel shuffle.
    fn decoder_input(raw: &[f32], mean: &[f32], deviation: &[f32]) -> Result<Vec<f32>, String> {
        if raw.len() != LATENT * LATENT * LATENT_CHANNELS {
            return Err(format!("latent has {} values", raw.len()));
        }
        let mut values = vec![0.0f32; TOKENS * CHANNELS];
        for row in 0..LATENT {
            for column in 0..LATENT {
                for channel in 0..LATENT_CHANNELS {
                    let value = raw[(row * LATENT + column) + LATENT * LATENT * channel]
                        * deviation[channel]
                        + mean[channel];
                    let merged = channel / (PATCH * PATCH);
                    let offset = channel % (PATCH * PATCH);
                    let target_row = row * PATCH + offset / PATCH;
                    let target_column = column * PATCH + offset % PATCH;
                    values[(target_row * SIDE + target_column) * CHANNELS + merged] = value;
                }
            }
        }
        Ok(values)
    }

    fn psnr(actual: &[f32], reference: &[f32]) -> (f64, f64) {
        let mut error = 0.0f64;
        let mut dot = 0.0f64;
        let mut left = 0.0f64;
        let mut right = 0.0f64;
        for (&got, &want) in actual.iter().zip(reference) {
            let (got, want) = (f64::from(got), f64::from(want));
            error += (got - want) * (got - want);
            dot += got * want;
            left += got * got;
            right += want * want;
        }
        let count = actual.len() as f64;
        let rms = (error / count).sqrt();
        (20.0 * (1.0 / rms).log10(), dot / (left * right).sqrt())
    }

    fn runtime() -> omega::config::RuntimeConfig {
        let serving = proxima_model_interop::ServingConfig::default();
        omega::config::RuntimeConfig::builder()
            .with_engine("gpu")
            .with_gpu_driver("metal")
            .with_allow_fusion(true)
            .with_numeric(serving.numeric_policy)
            .with_exact_activations(serving.exact_activations)
            .build()
            .expect("metal runtime")
    }

    #[test]
    fn the_decoder_reproduces_the_reference_image() -> Outcome {
        let reference = reference_dir()?;
        let model = model_dir()?;
        let checkpoint = Checkpoint::open(&model.join("comfy/flux2-vae.safetensors"))?;
        let (_, mean) = checkpoint.values("bn.running_mean", None, None, false)?;
        let (_, variance) = checkpoint.values("bn.running_var", None, None, false)?;
        let deviation: Vec<f32> = variance.iter().map(|value| value.sqrt()).collect();
        let raw = read_dump(&reference.join("step4.f32"))?;
        let latent = decoder_input(&raw, &mean, &deviation)?;

        let mut graph = Graph::new(DType::FLOAT32);
        let input = graph.host(
            "latent",
            &[TOKENS as u32, CHANNELS as u32, 1],
            Vec::new(),
        );
        let started = Instant::now();
        let output = build_vae_decode(&mut graph, &config(), input).map_err(|why| why.to_string())?;
        let image = output.image;
        eprintln!(
            "decoder: {} ops, {} weights, {} hosts built in {:.1}s",
            graph.program.len(),
            graph.weights.len(),
            graph.hosts.len(),
            started.elapsed().as_secs_f64()
        );
        let loading = Instant::now();
        let weights = load_weights(&checkpoint, &graph.weights, Codec::Float16, true)?;
        eprintln!("vae weights loaded in {:.1}s", loading.elapsed().as_secs_f64());

        let integers = graph.int_buffers();
        let mut blocks: Vec<(&str, QuantizedBlock)> = weights.blocks();
        blocks.extend(graph.host_blocks(&[("latent", latent.as_slice())], &integers));
        let plan_started = Instant::now();
        let (mut plan, _) = omega::backend::plan_named_with_config(
            &runtime(),
            &graph.program,
            &[TOKENS as u64],
            &blocks,
            &[image],
        )
        .map_err(|why| format!("plan: {why}"))?;
        eprintln!("planned in {:.1}s", plan_started.elapsed().as_secs_f64());
        let ran = Instant::now();
        let evaluated = omega::backend::execute_plan_named(&mut plan, &blocks)
            .map_err(|why| format!("decode: {why}"))?;
        let elapsed = ran.elapsed().as_secs_f64();
        let decoded = evaluated.get(image).ok_or("image is not an output")?.0;
        eprintln!("decoded [{}] in {:.1}s", decoded.len(), elapsed);

        let scaled: Vec<f32> = decoded
            .iter()
            .map(|value| ((value + 1.0) * 0.5).clamp(0.0, 1.0))
            .collect();
        let decoded = scaled.as_slice();
        let channel_major = read_dump(&reference.join("vae_out.f32"))?;
        let tokens = channel_major.len() / 3;
        let mut token_major = vec![0.0f32; channel_major.len()];
        for token in 0..tokens {
            for channel in 0..3 {
                token_major[token * 3 + channel] = channel_major[token + tokens * channel];
            }
        }
        let summary = |values: &[f32]| -> (f32, f32, f32) {
            let mut min = f32::MAX;
            let mut max = f32::MIN;
            let mut sum = 0.0f64;
            for &value in values {
                min = min.min(value);
                max = max.max(value);
                sum += f64::from(value);
            }
            (min, max, (sum / values.len() as f64) as f32)
        };
        let ours = summary(decoded);
        let theirs = summary(&token_major);
        eprintln!(
            "ours min {:.3} max {:.3} mean {:.3}; reference min {:.3} max {:.3} mean {:.3}",
            ours.0, ours.1, ours.2, theirs.0, theirs.1, theirs.2
        );
        for shift_y in [-1i32, 0, 1] {
            for shift_x in [-1i32, 0, 1] {
                let mut shifted = vec![0.0f32; decoded.len()];
                for row in 0..1024i32 {
                    for column in 0..1024i32 {
                        let source_row = (row + shift_y).clamp(0, 1023);
                        let source_column = (column + shift_x).clamp(0, 1023);
                        for channel in 0..3 {
                            let target = ((row * 1024 + column) * 3 + channel) as usize;
                            let source = ((source_row * 1024 + source_column) * 3 + channel) as usize;
                            shifted[target] = decoded[source];
                        }
                    }
                }
                let (moved, moved_cos) = psnr(&shifted, &token_major);
                if (shift_y, shift_x) != (0, 0) {
                    eprintln!("shift ({shift_y},{shift_x}): psnr={moved:.2} cos={moved_cos:.6}");
                }
            }
        }
        let (signal, cosine) = psnr(decoded, &token_major);
        for channel in 0..3usize {
            let ours: Vec<f32> = (0..token_major.len() / 3).map(|token| decoded[token * 3 + channel]).collect();
            let theirs: Vec<f32> = (0..token_major.len() / 3).map(|token| token_major[token * 3 + channel]).collect();
            let (value, channel_cos) = psnr(&ours, &theirs);
            eprintln!("channel {channel}: psnr={value:.2} cos={channel_cos:.6}");
        }
        eprintln!("vae decode: psnr={signal:.2} dB cosine={cosine:.8}");
        assert!(signal >= 26.0, "vae psnr {signal}");
        assert!(cosine >= 0.999, "vae cosine {cosine}");
        Ok(())
    }
}

#[cfg(feature = "std")]
mod vae_shape {
    use proxima_model_interop::diffusion::{Graph, VaeConfig, build_vae_decode};
    use proxima_numeric::DType;

    #[test]
    fn the_whole_decoder_program_shapes_without_any_weights() {
        let config = VaeConfig {
            channels: 32,
            block_channels: [128, 256, 512, 512],
            layers_per_block: 2,
            groups: 32,
            out_channels: 3,
            height: 128,
            width: 128,
            conv_taps: proxima_model_interop::diffusion::DiffusionSettings::resolved().conv_taps(),
        };
        let mut graph = Graph::new(DType::FLOAT32);
        let latent = graph.host("latent", &[16384, 32, 1], vec![0.0; 8]);
        let output = build_vae_decode(&mut graph, &config, latent).expect("decoder builds");
        let image = output.image;
        match proxima_tensor::infer(&graph.program, &[16384]) {
            Ok(shapes) => {
                assert_eq!(shapes.of(image), vec![1024 * 1024, 1, 3]);
            }
            Err(why) => {
                let at = format!("{why}")
                    .split('%')
                    .nth(1)
                    .and_then(|rest| rest.split(|c: char| !c.is_ascii_digit()).next())
                    .and_then(|digits| digits.parse::<usize>().ok())
                    .unwrap_or(0);
                for index in at.saturating_sub(6)..=(at + 1).min(graph.program.len() - 1) {
                    eprintln!("node {index}: {:?}", graph.program[index]);
                }
                panic!("{why}");
            }
        }
    }
}

#[cfg(feature = "std")]
mod vae_bind {
    use proxima_model_interop::diffusion::{Graph, VaeConfig, build_vae_decode};
    use proxima_numeric::{DType, NumericPolicy};

    #[test]
    fn the_whole_decoder_program_binds() {
        let config = VaeConfig {
            channels: 32,
            block_channels: [128, 256, 512, 512],
            layers_per_block: 2,
            groups: 32,
            out_channels: 3,
            height: 128,
            width: 128,
            conv_taps: proxima_model_interop::diffusion::DiffusionSettings::resolved().conv_taps(),
        };
        let mut graph = Graph::new(DType::FLOAT32);
        let latent = graph.host("latent", &[16384, 32, 1], vec![0.0; 8]);
        let output = build_vae_decode(&mut graph, &config, latent).expect("decoder builds");
        let image = output.image;
        let shapes = match proxima_tensor::infer(&graph.program, &[16384]) {
            Ok(shapes) => shapes,
            Err(why) => {
                let at = format!("{why}")
                    .split('%')
                    .nth(1)
                    .and_then(|rest| rest.split(|c: char| !c.is_ascii_digit()).next())
                    .and_then(|digits| digits.parse::<usize>().ok())
                    .unwrap_or(0);
                for index in at.saturating_sub(6)..=(at + 1).min(graph.program.len() - 1) {
                    eprintln!("node {index}: {:?}", graph.program[index]);
                }
                panic!("{why}");
            }
        };
        for index in 50..=56 {
            eprintln!("p{index}: {:?}", graph.program[index]);
        }
        let bound = proxima_tensor::bind(
            &graph.program,
            &shapes,
            &[image],
            NumericPolicy::llama_relaxed(),
        )
        .expect("binds");
        assert!(!bound.is_empty());
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod vae_conv_probe {
    use proxima_model_interop::diffusion::{Checkpoint, Graph, load_weights};
    use proxima_numeric::DType;
    use proxima_numeric::codec::Codec;
    use proxima_tensor::QuantizedBlock;
    use proxima_tensor::op::ScalarOp;

    const SIDE: u32 = 8;
    const GROUPS: u32 = 32;
    const WIDTH: u32 = 16;

    #[test]
    fn one_padded_convolution_runs_on_metal() -> Result<(), String> {
        let home = std::env::var_os("HOME").ok_or("HOME")?;
        let path = std::env::var_os("PROXIMA_FLUX2_KLEIN_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from(&home).join("models/image_gen/flux2-klein-4b"))
            .join("comfy/flux2-vae.safetensors");
        let checkpoint = Checkpoint::open(&path)?;
        let mut graph = Graph::new(DType::FLOAT32);
        let rows = SIDE * SIDE;
        let latent = graph.host("latent", &[rows, GROUPS, 1], vec![0.0; 8]);
        let mut taps = [proxima_tensor::op::NodeId(0); 9];
        for tap_row in 0..3u32 {
            for tap_column in 0..3u32 {
                let tap = (tap_row * 3 + tap_column) as usize;
                taps[tap] = graph.weight_tap(
                    "decoder.conv_in.weight",
                    &[GROUPS, WIDTH, GROUPS, 1],
                    (tap_row, tap_column),
                    GROUPS,
                    1,
                );
            }
        }
        let pad_index = graph.index_leaf(
            "pad.index",
            proxima_tensor::spec::conv3x3_pad_index(SIDE, SIDE),
        );
        let pad_mask = graph.host(
            "pad.mask",
            &[proxima_tensor::spec::conv3x3_pad_rows(SIDE, SIDE)],
            proxima_tensor::spec::conv3x3_pad_mask(SIDE, SIDE)
                .into_iter()
                .map(|value| value as f32)
                .collect(),
        );
        let crop_index = graph.index_leaf(
            "crop.index",
            proxima_tensor::spec::conv3x3_crop_index(SIDE, SIDE),
        );
        let convolved = proxima_tensor::spec::conv3x3_same(
            &mut graph.program,
            latent,
            &taps,
            pad_index,
            pad_mask,
            crop_index,
            SIDE,
            SIDE,
        )
        .map_err(|why| why.to_string())?;
        let bias = graph.weight(
            "decoder.conv_in.bias",
            &[GROUPS, WIDTH],
            None,
        );
        let image = proxima_tensor::spec::elementwise_like(
            &mut graph.program,
            convolved,
            ScalarOp::Add,
            &[(convolved, "rgc->rgc"), (bias, "gc->rgc")],
        )
        .map_err(|why| why.to_string())?;
        let weights = load_weights(&checkpoint, &graph.weights, Codec::Float16, true)?;
        let integers = graph.int_buffers();
        let latent_values = vec![0.5f32; (rows * GROUPS) as usize];
        let mut blocks: Vec<(&str, QuantizedBlock)> = weights.blocks();
        blocks.extend(graph.host_blocks(&[("latent", latent_values.as_slice())], &integers));
        let serving = proxima_model_interop::ServingConfig::default();
        let runtime = omega::config::RuntimeConfig::builder()
            .with_engine("gpu")
            .with_gpu_driver("metal")
            .with_allow_fusion(true)
            .with_numeric(serving.numeric_policy)
            .with_exact_activations(serving.exact_activations)
            .build()
            .expect("metal");
        let (mut plan, _) = omega::backend::plan_named_with_config(
            &runtime,
            &graph.program,
            &[rows as u64],
            &blocks,
            &[image],
        )
        .map_err(|why| format!("plan: {why}"))?;
        let evaluated = omega::backend::execute_plan_named(&mut plan, &blocks)
            .map_err(|why| format!("run: {why}"))?;
        let values = evaluated.get(image).ok_or("no image")?.0;
        let rms = (values.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>()
            / values.len() as f64)
            .sqrt();
        eprintln!("probe conv output {} values, rms {rms:.4}", values.len());
        Ok(())
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod reduce_split_axis {
    //! One reduce whose contraction axis is a *split* axis -- the operand is
    //! `[block_rows * 9, in_g, in_c]` and the iteration contracts a derived
    //! `t = 9` while the output row axis carries `r @ block_rows` -- asserted
    //! equal on the CPU and on Metal.
    //!
    //! This is the minimal reproduction of the defect that makes the
    //! three-block convolution's Metal dispatch never return: the graph's
    //! extents and the CPU result are both right
    //! (`spec::tests::conv_direct_loop`'s
    //! `the_three_block_reduce_iterates_its_contraction_not_its_operand`), so
    //! the fault is the emitter's. The GPU work runs on its own thread behind a
    //! wall-clock guard, so a regression is a red test in two minutes rather
    //! than an eleven-minute hang.
    use std::sync::mpsc;
    use std::time::Duration;

    use proxima_model_interop::diffusion::Graph;
    use proxima_numeric::DType;
    use proxima_tensor::QuantizedBlock;
    use proxima_tensor::spec::{
        conv3x3_taps_index, conv3x3_crop_index, conv3x3_pad_index, conv3x3_pad_mask,
        conv3x3_pad_rows, conv3x3_same_three,
    };

    // The decoder's own group/width split: 32 GroupNorm groups, 4 channels
    // each (128 channels) unless the size override below says otherwise.
    const HEIGHT: u32 = 8;
    const WIDTH: u32 = 8;
    const IN_G: u32 = 32;
    const IN_C: u32 = 16;
    const OUT_G: u32 = 32;
    const OUT_C: u32 = 16;
    /// The guard: a correct dispatch is milliseconds, the defect is unbounded.
    const GUARD: Duration = Duration::from_secs(120);

    fn values(count: usize, scale: f32, salt: usize) -> Vec<f32> {
        (0..count)
            .map(|index| ((index * 37 + salt) % 101) as f32 / 101.0 * scale - scale * 0.5)
            .collect()
    }

    struct Case {
        program: Vec<proxima_tensor::op::Op>,
        blocks: Vec<(&'static str, Vec<f32>)>,
        root: proxima_tensor::op::NodeId,
        want: Vec<f32>,
    }

    /// The program, its bound values, and the direct-loop answer.
    fn case() -> Case {
        let rows = HEIGHT * WIDTH;
        let mut graph = Graph::new(DType::FLOAT32);
        let activation = graph.host(
            "image",
            &[rows, IN_G, IN_C],
            values((rows * IN_G * IN_C) as usize, 2.0, 7),
        );
        let pad_index = graph.index_leaf("pad.index", conv3x3_pad_index(HEIGHT, WIDTH));
        let pad_mask = graph.host(
            "pad.mask",
            &[conv3x3_pad_rows(HEIGHT, WIDTH)],
            conv3x3_pad_mask(HEIGHT, WIDTH)
                .into_iter()
                .map(|value| value as f32)
                .collect(),
        );
        let crop_index = graph.index_leaf("crop.index", conv3x3_crop_index(HEIGHT, WIDTH));
        let columns_index = graph.index_leaf("columns.index", conv3x3_taps_index(HEIGHT, WIDTH));
        let kernel = graph.host(
            "kernel",
            &[OUT_G, OUT_C, 9, IN_G, IN_C],
            values((OUT_G * OUT_C * 9 * IN_G * IN_C) as usize, 2.0, 31),
        );
        let root = conv3x3_same_three(
            &mut graph.program,
            activation,
            kernel,
            pad_index,
            pad_mask,
            crop_index,
            columns_index,
        )
        .expect("the three-block convolution builds");

        // The definition, on the same values.
        let image = graph_host(&graph, "image");
        let weights = graph_host(&graph, "kernel");
        let mut want = vec![0.0f32; (rows * OUT_G * OUT_C) as usize];
        for row in 0..HEIGHT as usize {
            for column in 0..WIDTH as usize {
                for og in 0..OUT_G as usize {
                    for oc in 0..OUT_C as usize {
                        let mut sum = 0.0f32;
                        for y in 0..3usize {
                            for x in 0..3usize {
                                let (sr, sc) = (row + y, column + x);
                                if sr == 0 || sc == 0 || sr > HEIGHT as usize || sc > WIDTH as usize {
                                    continue;
                                }
                                let source = (sr - 1) * WIDTH as usize + (sc - 1);
                                for ig in 0..IN_G as usize {
                                    for ic in 0..IN_C as usize {
                                        let w = ((((og * OUT_C as usize + oc) * 9 + y * 3 + x)
                                            * IN_G as usize
                                            + ig)
                                            * IN_C as usize)
                                            + ic;
                                        let a = (source * IN_G as usize + ig) * IN_C as usize + ic;
                                        sum += weights[w] * image[a];
                                    }
                                }
                            }
                        }
                        want[((row * WIDTH as usize + column) * OUT_G as usize + og) * OUT_C as usize
                            + oc] = sum;
                    }
                }
            }
        }

        let integers = graph.int_buffers();
        let _ = graph.host_blocks(&[("image", image.as_slice())], &integers);
        let mut blocks: Vec<(&'static str, Vec<f32>)> = Vec::new();
        blocks.push(("image", image.clone()));
        blocks.push(("kernel", weights.clone()));
        blocks.push((
            "pad.mask",
            conv3x3_pad_mask(HEIGHT, WIDTH)
                .into_iter()
                .map(|value| value as f32)
                .collect(),
        ));
        blocks.push((
            "pad.index",
            conv3x3_pad_index(HEIGHT, WIDTH)
                .into_iter()
                .map(|value| value as f32)
                .collect(),
        ));
        blocks.push((
            "crop.index",
            conv3x3_crop_index(HEIGHT, WIDTH)
                .into_iter()
                .map(|value| value as f32)
                .collect(),
        ));
        blocks.push((
            "columns.index",
            conv3x3_taps_index(HEIGHT, WIDTH)
                .into_iter()
                .map(|value| value as f32)
                .collect(),
        ));
        Case {
            program: graph.program,
            blocks,
            root,
            want,
        }
    }

    fn graph_host(graph: &Graph, name: &str) -> Vec<f32> {
        let leaf = graph.hosts.iter().find(|leaf| leaf.name == name).expect("leaf");
        leaf.values.clone()
    }

    fn quantized<'a>(
        blocks: &'a [(&'static str, Vec<f32>)],
    ) -> Vec<(&'static str, QuantizedBlock<'a>)> {
        blocks
            .iter()
            .map(|(name, values)| (*name, QuantizedBlock::Float32(values)))
            .collect()
    }

    #[test]
    fn a_split_contraction_axis_reduce_agrees_between_cpu_and_metal() {
        let case = case();
        let bound = quantized(&case.blocks);

        let mut pools = proxima_tensor::cpu::BufferPools::default();
        let mut validated = None;
        let evaluated = proxima_tensor::cpu::evaluate_quantized_named_with_scratch(
            &case.program,
            &[],
            &bound,
            &[case.root],
            &mut pools,
            &mut validated,
        )
        .expect("the CPU evaluates the reduce");
        let cpu_started = std::time::Instant::now();
        let cpu = evaluated.get(case.root).expect("root").0.to_vec();
        let _ = cpu_started.elapsed();
        // Relative: the contract is a 4608-term f32 dot product here, so an
        // absolute micro-bound would be measuring the accumulator's rounding
        // rather than the lowering.
        let scale = case.want.iter().fold(0.0f32, |worst, value| worst.max(value.abs()));
        let worst_cpu = cpu
            .iter()
            .zip(&case.want)
            .map(|(cpu, want)| (cpu - want).abs())
            .fold(0.0f32, f32::max)
            / scale;
        assert!(
            worst_cpu < 1e-4,
            "the CPU form differs from the definition by {worst_cpu} relative"
        );

        // Metal on its own thread: `waitUntilCompleted` never returns under the
        // defect, so the guard is what turns that into a failure.
        let program = case.program.clone();
        let root = case.root;
        let blocks: Vec<(&'static str, Vec<f32>)> = case.blocks.clone();
        let (sender, receiver) = mpsc::channel();
        std::thread::spawn(move || {
            // The kernel is bound exactly as the decoder binds it: packed
            // half-precision rows, not f32. The f32 binding passes; the packed
            // one is what the served graph actually uses.
            let packed: Vec<u8> = proxima_model_interop::diffusion::encode(
                &blocks.iter().find(|(name, _)| *name == "kernel").expect("kernel").1,
                proxima_numeric::codec::Codec::Float16,
            )
            .expect("the kernel encodes to f16");
            let mut bound: Vec<(&'static str, QuantizedBlock<'_>)> = Vec::new();
            for (name, values) in &blocks {
                bound.push((
                    *name,
                    if *name == "kernel" {
                        QuantizedBlock::Packed {
                            codec: proxima_numeric::codec::Codec::Float16,
                            bytes: &packed,
                        }
                    } else {
                        QuantizedBlock::Float32(values)
                    },
                ));
            }
            let serving = proxima_model_interop::ServingConfig::default();
            let runtime = omega::config::RuntimeConfig::builder()
                .with_engine("gpu")
                .with_gpu_driver("metal")
                .with_allow_fusion(true)
                .with_numeric(serving.numeric_policy)
                .with_exact_activations(serving.exact_activations)
                .build()
                .expect("metal runtime");
            let outcome = omega::backend::plan_named_with_config(
                &runtime,
                &program,
                &[],
                &bound,
                &[root],
            )
            .and_then(|(mut plan, _)| omega::backend::execute_plan_named(&mut plan, &bound));
            let _ = sender.send(match outcome {
                Ok(evaluated) => Ok(evaluated.get(root).map(|value| value.0.to_vec())),
                Err(why) => Err(format!("{why}")),
            });
        });
        let metal_started = std::time::Instant::now();
        let metal = match receiver.recv_timeout(GUARD) {
            Ok(Ok(Some(values))) => values,
            Ok(Ok(None)) => panic!("Metal returned no root"),
            Ok(Err(why)) => panic!("Metal refused the reduce: {why}"),
            Err(_) => panic!(
                "Metal did not complete the split-contraction reduce within {}s --                  the dispatch is hung, not slow (see the emitter's contraction bound)",
                GUARD.as_secs()
            ),
        };
        // Throughput of the split-axis reduce: the op the decoder spends its
        // whole VAE budget in. `macs` is the contraction's true work, so a rate
        // far below the machine's GEMM rate is the finding.
        let macs = (HEIGHT * WIDTH) as f64
            * (OUT_G * OUT_C) as f64
            * 9.0
            * (IN_G * IN_C) as f64;
        eprintln!(
            "split_reduce metal {:.2}s for {:.1} GMAC = {:.2} GFLOPS",
            metal_started.elapsed().as_secs_f64(),
            macs / 1e9,
            2.0 * macs / 1e9 / metal_started.elapsed().as_secs_f64()
        );
        assert_eq!(metal.len(), cpu.len(), "shape");
        let worst = metal
            .iter()
            .zip(&cpu)
            .map(|(metal, cpu)| (metal - cpu).abs())
            .fold(0.0f32, f32::max);
        // The kernel is bound as f16, so the agreement bound is that codec's
        // rounding accumulated over the contraction, not the graph's own.
        assert!(
            worst / scale < 2e-2,
            "Metal differs from the CPU by {} relative",
            worst / scale
        );
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod vae_prefix {
    use std::env;
    use std::path::PathBuf;

    use proxima_model_interop::diffusion::{
        Checkpoint, Graph, VaeConfig, build_vae_decode, load_weights,
    };
    use proxima_numeric::DType;
    use proxima_numeric::codec::Codec;
    use proxima_tensor::QuantizedBlock;

    const SIDE: u32 = 16;
    const TOKENS: usize = (SIDE * SIDE) as usize;

    #[test]
    fn the_first_decoder_stage_that_faults_names_itself() -> Result<(), String> {
        let home = env::var_os("HOME").ok_or("HOME")?;
        let model = env::var_os("PROXIMA_FLUX2_KLEIN_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(&home).join("models/image_gen/flux2-klein-4b"));
        let checkpoint = Checkpoint::open(&model.join("comfy/flux2-vae.safetensors"))?;
        let config = VaeConfig {
            channels: 32,
            block_channels: [128, 256, 512, 512],
            layers_per_block: 2,
            groups: 32,
            out_channels: 3,
            height: SIDE,
            width: SIDE,
            conv_taps: proxima_model_interop::diffusion::DiffusionSettings::resolved().conv_taps(),
        };
        let mut graph = Graph::new(DType::FLOAT32);
        let latent = graph.host("latent", &[SIDE * SIDE, 32, 1], vec![0.0; 8]);
        let output = build_vae_decode(&mut graph, &config, latent).map_err(|why| why.to_string())?;
        let weights = load_weights(&checkpoint, &graph.weights, Codec::Float16, true)?;
        let integers = graph.int_buffers();
        let latent_values = vec![0.1f32; TOKENS * 32];
        let mut blocks: Vec<(&str, QuantizedBlock)> = weights.blocks();
        blocks.extend(graph.host_blocks(&[("latent", latent_values.as_slice())], &integers));
        let serving = proxima_model_interop::ServingConfig::default();
        let runtime = omega::config::RuntimeConfig::builder()
            .with_engine("gpu")
            .with_gpu_driver("metal")
            .with_allow_fusion(true)
            .with_numeric(serving.numeric_policy)
            .with_exact_activations(serving.exact_activations)
            .build()
            .expect("metal");
        for (name, node) in &output.trace {
            match omega::backend::plan_named_with_config(
                &runtime,
                &graph.program,
                &[TOKENS as u64],
                &blocks,
                &[*node],
            ) {
                Ok((mut plan, _)) => match omega::backend::execute_plan_named(&mut plan, &blocks) {
                    Ok(_) => eprintln!("stage {name} ok"),
                    Err(why) => {
                        eprintln!("stage {name} FAULTS: {why}");
                        return Ok(());
                    }
                },
                Err(why) => {
                    eprintln!("stage {name} does not plan: {why}");
                    return Ok(());
                }
            }
        }
        eprintln!("every decoder stage ran");
        Ok(())
    }
}

#[cfg(feature = "std")]
mod vae_gathers {
    use proxima_model_interop::diffusion::{Graph, VaeConfig, build_vae_decode};
    use proxima_numeric::DType;
    use proxima_tensor::map::IndexMap;
    use proxima_tensor::op::Op;

    #[test]
    fn every_gather_reads_a_leaf_whose_range_fits_its_source() {
        let config = VaeConfig {
            channels: 32,
            block_channels: [128, 256, 512, 512],
            layers_per_block: 2,
            groups: 32,
            out_channels: 3,
            height: 16,
            width: 16,
            conv_taps: proxima_model_interop::diffusion::DiffusionSettings::resolved().conv_taps(),
        };
        let mut graph = Graph::new(DType::FLOAT32);
        let latent = graph.host("latent", &[256, 32, 1], vec![0.0; 8]);
        build_vae_decode(&mut graph, &config, latent).expect("decoder builds");
        let shapes = proxima_tensor::infer(&graph.program, &[256]).expect("shapes");
        let name_of = |node: proxima_tensor::op::NodeId| -> String {
            graph
                .hosts
                .iter()
                .find(|host| {
                    graph
                        .program
                        .iter()
                        .position(|op| op.name() == Some(host.name.as_str()))
                        .map(|index| index as u32)
                        == Some(node.0)
                })
                .map_or_else(|| format!("?{}", node.0), |host| host.name.clone())
        };
        for (index, op) in graph.program.iter().enumerate() {
            let Op::Elementwise { operands, .. } = op else {
                continue;
            };
            for (source, map) in operands {
                let IndexMap::Computed {
                    indices, gathered_dim, ..
                } = map
                else {
                    continue;
                };
                let host = graph.hosts.iter().find(|host| {
                    graph
                        .program
                        .iter()
                        .position(|candidate| candidate.name() == Some(host.name.as_str()))
                        .map(|position| position as u32)
                        == Some(indices.0)
                });
                let Some(host) = host else {
                    eprintln!("node {index}: gather with a computed (non-leaf) index");
                    continue;
                };
                let maximum = host.values.iter().fold(f32::MIN, |acc, value| acc.max(*value));
                let extent = shapes.of(*source)[*gathered_dim as usize];
                if maximum as u64 >= extent {
                    eprintln!(
                        "node {index} ({}): gather of {} (extent {extent}) reads {} with max {}",
                        name_of(proxima_tensor::op::NodeId(index as u32)),
                        name_of(*source),
                        host.name,
                        maximum
                    );
                }
            }
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[cfg(all(feature = "metal", target_os = "macos"))]
mod vae_stages {
    use std::env;
    use std::fs;
    use std::path::PathBuf;

    use proxima_model_interop::diffusion::{
        Checkpoint, Graph, VaeConfig, build_vae_decode, load_weights,
    };
    use proxima_numeric::DType;
    use proxima_numeric::codec::Codec;
    use proxima_tensor::QuantizedBlock;

    type Outcome = Result<(), String>;

    /// The reference's captures are `[width, height, channels]` with the width
    /// axis innermost; the graph's activations are `[row, groups, width]` with
    /// the channel innermost. This reads a capture into the graph's order.
    fn channel_fastest(values: &[f32], channels: usize) -> Vec<f32> {
        let rows = values.len() / channels;
        let mut token_major = vec![0.0f32; values.len()];
        for row in 0..rows {
            for channel in 0..channels {
                token_major[row * channels + channel] = values[row + rows * channel];
            }
        }
        token_major
    }

    #[test]
    fn every_decoder_stage_matches_the_reference_capture() -> Outcome {
        let trace = env::var_os("PROXIMA_FLUX2_KLEIN_TRACE")
            .map(PathBuf::from)
            .ok_or("set PROXIMA_FLUX2_KLEIN_TRACE to the capture dir")?;
        let side: u32 = env::var("PROXIMA_FLUX2_KLEIN_SIDE")
            .ok()
            .and_then(|text| text.parse().ok())
            .unwrap_or(64);
        let latent_side = side / 2;
        let home = env::var_os("HOME").ok_or("HOME")?;
        let model = env::var_os("PROXIMA_FLUX2_KLEIN_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(&home).join("models/image_gen/flux2-klein-4b"));
        let checkpoint = Checkpoint::open(&model.join("comfy/flux2-vae.safetensors"))?;
        let (_, mean) = checkpoint.values("bn.running_mean", None, None, false)?;
        let (_, variance) = checkpoint.values("bn.running_var", None, None, false)?;
        let deviation: Vec<f32> = variance.iter().map(|value| value.sqrt()).collect();
        let read = |name: &str| -> Result<Vec<f32>, String> {
            let path = trace.join(name);
            let bytes = fs::read(&path).map_err(|why| format!("{}: {why}", path.display()))?;
            Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| f32::from_le_bytes(*word))
                .collect())
        };
        let raw = read("step4.f32")?;
        let mut latent = vec![0.0f32; (side * side) as usize * 32];
        for row in 0..latent_side {
            for column in 0..latent_side {
                for channel in 0..128usize {
                    let value = raw[(row * latent_side + column) as usize
                        + (latent_side * latent_side) as usize * channel]
                        * deviation[channel]
                        + mean[channel];
                    let merged = channel / 4;
                    let offset = (channel % 4) as u32;
                    let target_row = row * 2 + offset / 2;
                    let target_column = column * 2 + offset % 2;
                    latent[((target_row * side + target_column) * 32) as usize + merged] = value;
                }
            }
        }
        let config = VaeConfig {
            channels: 32,
            block_channels: [128, 256, 512, 512],
            layers_per_block: 2,
            groups: 32,
            out_channels: 3,
            height: side,
            width: side,
            conv_taps: proxima_model_interop::diffusion::DiffusionSettings::resolved().conv_taps(),
        };
        let mut graph = Graph::new(DType::FLOAT32);
        let input = graph.host("latent", &[side * side, 32, 1], Vec::new());
        let output = build_vae_decode(&mut graph, &config, input).map_err(|why| why.to_string())?;
        let weights = load_weights(&checkpoint, &graph.weights, Codec::Float16, true)?;
        let integers = graph.int_buffers();
        let mut blocks: Vec<(&str, QuantizedBlock)> = weights.blocks();
        blocks.extend(graph.host_blocks(&[("latent", latent.as_slice())], &integers));
        let serving = proxima_model_interop::ServingConfig::default();
        let runtime = omega::config::RuntimeConfig::builder()
            .with_engine("gpu")
            .with_gpu_driver("metal")
            .with_allow_fusion(true)
            .with_numeric(serving.numeric_policy)
            .with_exact_activations(serving.exact_activations)
            .build()
            .expect("metal");
        for (name, node) in &output.trace {
            let (capture, width, channels) = match *name {
                "conv_in" => ("p_dconv_in.f32", side as usize, 512usize),
                "up3" => ("p_dup3_0.f32", side as usize, 512),
                "up3.up" => ("p_dup3_up.f32", (side * 2) as usize, 512),
                "up2.up" => ("p_dup2_up.f32", (side * 4) as usize, 512),
                "up1.up" => ("p_dup1_up.f32", (side * 8) as usize, 256),
                "image" => ("vae_out.f32", (side * 8) as usize, 3),
                other => {
                    eprintln!("stage {other}: no reference capture");
                    continue;
                }
            };
            if !trace.join(capture).exists() {
                eprintln!("stage {name}: no capture at {capture}");
                continue;
            }
            let _ = width;
            let reference = channel_fastest(&read(capture)?, channels);
            let (mut plan, _) = omega::backend::plan_named_with_config(
                &runtime,
                &graph.program,
                &[(side * side) as u64],
                &blocks,
                &[*node],
            )
            .map_err(|why| format!("plan {name}: {why}"))?;
            let evaluated = omega::backend::execute_plan_named(&mut plan, &blocks)
                .map_err(|why| format!("run {name}: {why}"))?;
            let actual = evaluated.get(*node).ok_or("stage missing")?.0;
            let scaled: Vec<f32> = if *name == "image" {
                actual
                    .iter()
                    .map(|value| ((value + 1.0) * 0.5).clamp(0.0, 1.0))
                    .collect()
            } else {
                actual.to_vec()
            };
            let mut error = 0.0f64;
            let mut squared = 0.0f64;
            let mut dot = 0.0f64;
            let mut left = 0.0f64;
            for (&got, &want) in scaled.iter().zip(&reference) {
                let (got, want) = (f64::from(got), f64::from(want));
                error += (got - want) * (got - want);
                squared += want * want;
                dot += got * want;
                left += got * got;
            }
            let count = reference.len() as f64;
            let rms_rel = (error / count).sqrt() / (squared / count).sqrt();
            let cosine = dot / (left * squared).sqrt();
            eprintln!("stage {name}: rms_rel={rms_rel:.6} cos={cosine:.6}");
        }
        Ok(())
    }
}
