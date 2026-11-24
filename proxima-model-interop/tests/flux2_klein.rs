//! Acceptance tests for the flux2-klein spec (`specs/flux2-klein/SPEC.md`). The library code
//! under test is generic; the reference dumps are what name the model.

/// The kernel extent the checkpoint-free shape tests give `VaeConfig::window`.
/// `DiffusionSettings` carries no window; the real value is read from the
/// checkpoint (`Checkpoint::window_extent("decoder.conv_in.weight")`, as the
/// gated decode tests do), and the flux2-klein decoder's convolutions are 3x3,
/// so shape-only tests use that extent -- the one place it is spelled.
#[cfg(feature = "std")]
fn shape_only_window() -> u32 {
    3
}

/// Decodes a `PackedRows` into host elements; the `?` returns the caller's `String` error.
macro_rules! decoded {
    ($rows:expr) => {{
        let mut values = Vec::new();
        $rows.decode_all(&mut values).map_err(|why| why.to_string())?;
        values
    }};
}

/// A measurement `compare` returned, decoded; the `?` returns the caller's `String` error.
macro_rules! measured {
    ($result:expr) => {{
        let rows = $result.map_err(|why| why.to_string())?;
        decoded!(rows)
    }};
}

/// Host-element rows over a slice of decoded values.
macro_rules! host_rows {
    ($values:expr) => {
        ::proxima_numeric::PackedRows::from_values(
            &::proxima_numeric::NumericFormat::host_element(),
            $values,
        )
        .map_err(|why| why.to_string())?
    };
}

/// `(rms_rel, cosine)` of two runs of decoded values.
#[cfg(all(feature = "metal", target_os = "macos"))]
macro_rules! stats {
    ($actual:expr, $reference:expr) => {{
        let (actual, reference) = (host_rows!($actual), host_rows!($reference));
        let rms = measured!(::proxima_numeric::compare::rms_rel(&actual, &reference));
        let cosine = measured!(::proxima_numeric::compare::cosine(&actual, &reference));
        (rms[0], cosine[0])
    }};
}

/// `(psnr, cosine)` of two runs of decoded values on a unit full scale.
#[cfg(all(feature = "metal", target_os = "macos"))]
macro_rules! fidelity {
    ($actual:expr, $reference:expr) => {{
        let (actual, reference) = (host_rows!($actual), host_rows!($reference));
        let full_scale = host_rows!(&[1.0]);
        let signal = measured!(::proxima_numeric::compare::psnr(&actual, &reference, &full_scale));
        let cosine = measured!(::proxima_numeric::compare::cosine(&actual, &reference));
        (signal[0], cosine[0])
    }};
}

/// `(min, max, mean)` of decoded values.
#[cfg(all(feature = "metal", target_os = "macos"))]
macro_rules! summary {
    ($values:expr) => {{
        let values = $values;
        let low = values.iter().copied().reduce(|low, value| low.min(value)).ok_or("no values")?;
        let high = values.iter().copied().reduce(|high, value| high.max(value)).ok_or("no values")?;
        let moments = measured!(::proxima_numeric::compare::mean_std(&host_rows!(values)));
        (low, high, moments[0])
    }};
}

/// A reference dump decoded to host elements; the `?` returns the caller's `String` error.
#[cfg(all(feature = "metal", target_os = "macos"))]
macro_rules! dump_values {
    ($path:expr) => {{
        let rows = crate::oracle::read_dump($path)?;
        decoded!(rows)
    }};
}

/// The dump `$key` names in `data/flux2_oracle.toml`, found under `$dir`, decoded.
#[cfg(all(feature = "metal", target_os = "macos"))]
macro_rules! named_values {
    ($dir:expr, $key:expr) => {
        dump_values!(&$dir.join(crate::oracle::file("dumps", $key)?))
    };
}

/// Channel-major values (`index = token + tokens * channel`) as token-major; a reshape, so any
/// element type.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn token_major_and<T: Copy>(channel_major: &[T], tokens: usize, channels: usize) -> Vec<T> {
    (0..tokens * channels)
        .map(|index| channel_major[(index / channels) + tokens * (index % channels)])
        .collect()
}

/// The reference dumps: their names and the parity tolerances come from
/// `data/flux2_oracle.toml`; their element format is `DiffusionSettings::reference_format`.
mod oracle {
    use std::fmt::Display;
    #[cfg(feature = "std")]
    use std::fs;
    #[cfg(feature = "std")]
    use std::path::Path;
    use std::str::FromStr;

    #[cfg(feature = "std")]
    use proxima_model_interop::diffusion::DiffusionSettings;
    #[cfg(feature = "std")]
    use proxima_numeric::PackedRows;
    use proxima_numeric::{FloatLayout, NumericFormat};

    const ORACLE: &str = include_str!("data/flux2_oracle.toml");
    const FALLBACK: &str = "*";

    fn table() -> Result<toml::Table, String> {
        ORACLE.parse().map_err(|why: toml::de::Error| why.to_string())
    }

    /// Whether `measured` is within the `[tolerances]` entry `key`, read in `measured`'s type.
    pub fn within<T: PartialOrd + FromStr>(measured: T, key: &str) -> Result<bool, String>
    where
        T::Err: Display,
    {
        let bound: T = file("tolerances", key)?
            .parse()
            .map_err(|why: T::Err| format!("flux2_oracle.toml [tolerances] {key}: {why}"))?;
        Ok(measured <= bound)
    }

    pub fn file(section: &str, key: &str) -> Result<String, String> {
        let table = table()?;
        let entries = table
            .get(section)
            .and_then(toml::Value::as_table)
            .ok_or_else(|| format!("flux2_oracle.toml has no [{section}] section"))?;
        let direct = entries.get(key).and_then(toml::Value::as_str).map(str::to_owned);
        let pattern = entries
            .get(FALLBACK)
            .and_then(toml::Value::as_str)
            .map(|template| template.replace("{name}", key));
        direct
            .or(pattern)
            .ok_or_else(|| format!("flux2_oracle.toml [{section}] names no file for {key}"))
    }

    #[cfg(feature = "std")]
    pub fn read_dump(path: &Path) -> Result<PackedRows, String> {
        let format = DiffusionSettings::try_resolved()
            .map_err(|why| why.to_string())?
            .reference_format;
        let bytes = fs::read(path).map_err(|why| format!("{}: {why}", path.display()))?;
        PackedRows::from_bytes(&format, bytes)
            .map_err(|why| format!("{}: {why}", path.display()))
    }

    pub fn host_layout() -> Result<FloatLayout, String> {
        NumericFormat::host_element()
            .element_layout()
            .ok_or_else(|| String::from("the host element is a scalar format"))
    }
}

#[cfg(feature = "std")]
mod activation {
    use proxima_model_interop::diffusion::DiffusionSettings;
    use proxima_numeric::DType;

    pub fn dit() -> Result<DType, String> {
        DiffusionSettings::try_resolved()
            .map_err(|why| why.to_string())?
            .dit_activation_dtype()
    }

    pub fn vae() -> Result<DType, String> {
        DiffusionSettings::try_resolved()
            .map_err(|why| why.to_string())?
            .vae_activation_dtype()
    }
}

#[cfg(feature = "std")]
mod philox {
    use std::env;
    use std::path::PathBuf;

    use proxima_numeric::{NumericFormat, PackedRows, Philox, RngError, RngKind, compare};

    const SEED: u64 = 42;
    const LATENT_VALUES: usize = 64 * 64 * 128;
    const HEAD: usize = 16;

    type Outcome = Result<(), String>;

    fn reference_dir() -> Result<PathBuf, String> {
        if let Some(dir) = env::var_os("PROXIMA_FLUX2_KLEIN_REF") {
            return Ok(PathBuf::from(dir));
        }
        let home = env::var_os("HOME").ok_or("HOME locates the default reference dir")?;
        Ok(PathBuf::from(home).join(".cache/kvgate/proxnum588/ref"))
    }

    fn reference_noise() -> Result<PackedRows, String> {
        crate::oracle::read_dump(&reference_dir()?.join(crate::oracle::file("dumps", "noise")?))
    }

    fn ours(format: &NumericFormat) -> Result<PackedRows, String> {
        let selected: RngKind = "philox".parse().map_err(|why: RngError| why.to_string())?;
        PackedRows::from_values(format, &selected.generator(SEED).normal(LATENT_VALUES))
            .map_err(|why| why.to_string())
    }

    fn head(rows: &PackedRows) -> Result<PackedRows, String> {
        let mut kept = rows.clone();
        kept.truncate(HEAD).map_err(|why| why.to_string())?;
        Ok(kept)
    }

    #[test]
    fn the_first_sixteen_values_of_seed_42_match_the_reference_dump() -> Outcome {
        let reference = reference_noise()?;
        let generated = ours(reference.format())?;
        let gap = compare::max_abs_diff(&head(&generated)?, &head(&reference)?);
        let worst = measured!(gap)[0];
        assert!(crate::oracle::within(worst, "philox")?, "max abs diff {worst}");
        Ok(())
    }

    #[test]
    fn all_524288_values_of_seed_42_match_the_reference_dump() -> Outcome {
        let reference = reference_noise()?;
        let generated = ours(reference.format())?;
        assert_eq!(reference.elements(), LATENT_VALUES);
        assert_eq!(generated.elements(), LATENT_VALUES);
        let worst = measured!(compare::max_abs_diff(&generated, &reference))[0];
        let (ours_values, reference_values) = (decoded!(generated), decoded!(reference));
        println!(
            "max abs diff {worst}; ours {:?}; ref {:?}",
            &ours_values[..4],
            &reference_values[..4]
        );
        assert!(crate::oracle::within(worst, "philox")?, "max abs diff {worst}");
        Ok(())
    }

    #[test]
    fn the_latent_is_standard_normal_and_unsupported_generators_are_refused_by_name() -> Outcome {
        let generated = Philox::new(SEED).normal(LATENT_VALUES);
        let moments = measured!(compare::mean_std(&host_rows!(&generated)));
        let (mean, spread) = (moments[0], moments[1]);
        assert!((-0.01..=0.01).contains(&mean), "mean {mean}");
        assert!((0.99..=1.01).contains(&spread), "std {spread}");
        for name in ["std_default", "cpu"] {
            let refused = name.parse::<RngKind>();
            assert_eq!(refused, Err(RngError::Unimplemented { name }));
        }
        Ok(())
    }
}

mod scheduler {
    use std::env;
    use std::fs;
    use std::path::PathBuf;

    use std::num::ParseFloatError;

    use proxima_numeric::{FlowMatchSchedule, PackedRows, StepRule, euler_step, step};

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

    fn reference_sigmas() -> Result<PackedRows, String> {
        let path = reference_dir()?.join("sigmas.txt");
        let text = fs::read_to_string(&path)
            .map_err(|why| format!("fixture {}: {why}", path.display()))?;
        let parse = |word: &str| word.parse().map_err(|why: ParseFloatError| format!("{word}: {why}"));
        let values = text.split_whitespace().map(parse).collect::<Result<Vec<_>, String>>()?;
        Ok(host_rows!(&values))
    }

    #[test]
    fn sigmas_for_4096_tokens_and_4_steps_match_the_reference_log() -> Outcome {
        let reference = decoded!(reference_sigmas()?);
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
        let start = [0.5, -1.25, 3.0, 0.0, 7.5];
        let velocity = [1.0, -2.0, 0.25, 4.0, -0.5];
        let (sigma, sigma_next) = (0.967_384_04, 0.908_143_94);
        let layout = super::oracle::host_layout().expect("the host element has a layout");
        let mut latent = start;
        euler_step(&mut latent, &velocity, sigma, sigma_next, layout)
            .expect("the host element steps");
        let mut through_rule = start;
        step(StepRule::Euler, &mut through_rule, &velocity, sigma, sigma_next, layout)
            .expect("the host element steps");
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
    use std::path::PathBuf;
    #[cfg(feature = "instrument")]
    use std::time::Duration;
    use std::time::Instant;

    use conflaguration::Settings;
    use proxima_model_interop::ServingSettings;
    use proxima_model_interop::diffusion::{
        Checkpoint, Denoiser, DiffusionSettings, DitConfig, DitInputs, Graph, attention_scale,
        build_dit, head_norm_count,
    };
    use proxima_numeric::{FlowMatchSchedule, PackedRows, euler_step};
    use proxima_tensor::op::NodeId;
    use proxima_tensor::spec::multi_axis_rope_tables;

    const IMAGE_TOKENS: usize = 64 * 64;
    const TEXT_TOKENS: usize = 512;
    const HIDDEN: usize = 128;
    const JOINT: usize = 4608;
    const STEPS: u32 = 4;
    const ROPE_DAXES: [u32; 4] = [32, 32, 32, 32];
    const ROPE_THETA: u16 = 2000;

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

    /// The reference latent is channel-major (`index = token + tokens * channel`);
    /// the graph reads it token-major.
    fn token_major<T: Copy>(channel_major: &[T]) -> Vec<T> {
        super::token_major_and(channel_major, IMAGE_TOKENS, HIDDEN)
    }

    /// A `[channels, tokens]` capture (dim0 the channel axis) as token-major.
    fn from_channel_major<T: Copy>(values: &[T], tokens: usize, channels: usize) -> Vec<T> {
        (0..tokens * channels)
            .map(|index| values[(index % channels) + channels * (index / channels)])
            .collect()
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
            rope_theta: ROPE_THETA.into(),
        }
    }

    fn rope_positions<T: Copy + From<u16>>() -> Result<Vec<T>, String> {
        let mut positions = Vec::with_capacity(JOINT * 4);
        let zero = T::from(0);
        let text_tokens = u16::try_from(TEXT_TOKENS).map_err(|why| why.to_string())?;
        for token in 0..text_tokens {
            positions.extend_from_slice(&[zero, zero, zero, token.into()]);
        }
        let width: u16 = 64;
        let image_tokens = u16::try_from(IMAGE_TOKENS).map_err(|why| why.to_string())?;
        for token in 0..image_tokens {
            let (row, column) = (token / width, token % width);
            positions.extend_from_slice(&[zero, row.into(), column.into(), zero]);
        }
        Ok(positions)
    }

    fn serving_settings() -> Result<ServingSettings, String> {
        ServingSettings::from_env().map_err(|why| why.to_string())
    }

    #[test]
    fn velocity_and_the_four_euler_steps_match_the_reference() -> Outcome {
        let fixtures = fixtures()?;
        let started = Instant::now();
        let activation_dtype = super::activation::dit()?;
        let mut graph = Graph::new(activation_dtype).expect("diffusion settings validate");
        let latent = graph.host("latent", &[IMAGE_TOKENS as u32, HIDDEN as u32], graph.unbound());
        let text = graph.host("text", &[TEXT_TOKENS as u32, 7680], graph.unbound());
        let timestep = graph.host("timestep", &[256], graph.unbound());
        let (cos_bytes, sin_bytes) = multi_axis_rope_tables(
            &rope_positions()?,
            &ROPE_DAXES,
            &[ROPE_THETA.into(); 4],
            activation_dtype,
        )
        .map_err(|why| why.to_string())?;
        let cos_rows = PackedRows::from_bytes(graph.activation_format(), cos_bytes);
        let sin_rows = PackedRows::from_bytes(graph.activation_format(), sin_bytes);
        let rope_cos = graph.host("rope_cos", &[JOINT as u32, 64], cos_rows);
        let rope_sin = graph.host("rope_sin", &[JOINT as u32, 64], sin_rows);
        let config = dit_config();
        let norm_count = head_norm_count(&mut graph, &config);
        let scale = attention_scale(&mut graph, &config)
            .expect("the attention scale builds in the program");
        let inputs = DitInputs {
            latent,
            text,
            timestep,
            rope_cos,
            rope_sin,
            norm_count,
            attention_scale: scale,
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
        eprintln!(
            "program: {} ops, {} weight leaves, {} host leaves, built in {:?}",
            graph.program.len(),
            graph.weights.len(),
            graph.hosts.len(),
            started.elapsed()
        );

        let checkpoint = Checkpoint::open(&fixtures.model.join("flux-2-klein-4b.safetensors"))?;
        let diffusion = DiffusionSettings::try_resolved().map_err(|why| why.to_string())?;
        let activation = graph.activation();
        let extra: Vec<NodeId> = output.trace.iter().map(|(_, node)| *node).collect();
        let loading = Instant::now();
        let mut denoiser = Denoiser::new(
            &checkpoint,
            graph,
            velocity,
            &[JOINT as u64],
            diffusion.dit_weights,
            &serving_settings()?.as_serving_config(&[]),
        )?;
        eprintln!("weights loaded in {:?}", loading.elapsed());

        let mut latent_values = token_major(&named_values!(fixtures.reference, "noise"));
        let text_values = named_values!(fixtures.reference, "text");
        let layout = super::oracle::host_layout()?;
        let sigmas = FlowMatchSchedule::new(STEPS, IMAGE_TOKENS as u32)
            .sigmas()
            .map_err(|why| why.to_string())?;

        let mut call_graph = Graph::new(activation).expect("diffusion settings validate");
        call_graph.host("latent", &[IMAGE_TOKENS as u32, HIDDEN as u32], call_graph.unbound());
        call_graph.host("text", &[TEXT_TOKENS as u32, 7680], call_graph.unbound());
        call_graph.host("timestep", &[256], call_graph.unbound());

        let reference_velocity = token_major(&named_values!(fixtures.reference, "velocity1"));
        let mut verdicts = 0;
        for step in 0..STEPS as usize {
            let settings = DiffusionSettings::try_resolved().map_err(|why| why.to_string())?;
            let format = settings.dit_activation;
            let timestep_rows = PackedRows::from_values(&format, &[sigmas[step]])
                .map_err(|why| why.to_string())
                .and_then(|sigma| settings.timestep_embedding(&sigma, 256, &format))
                .expect("the timestep embeds");
            let latent_rows = PackedRows::from_values(call_graph.activation_format(), &latent_values)
                .map_err(|why| why.to_string())?;
            let text_rows = PackedRows::from_values(call_graph.activation_format(), &text_values)
                .map_err(|why| why.to_string())?;
            let ran = Instant::now();
            let named = call_graph.host_blocks(&[
                ("latent", &latent_rows),
                ("text", &text_rows),
                ("timestep", &timestep_rows),
            ])?;
            let named = named.as_slice();
            let evaluated = {
            #[cfg(feature = "instrument")]
            let profiled = if env::var_os("PROXIMA_FLUX2_KLEIN_PROFILE").is_some() {
                let (evaluated, timings) = denoiser
                    .evaluate_op_timed(named)
                    .map_err(|why| format!("step {}: {why}", step + 1))?;
                let mut rows: Vec<(&str, u64, Vec<u64>)> = timings
                    .iter()
                    .map(|timing| (timing.kind, timing.gpu_ns, timing.extents.clone()))
                    .collect();
                rows.sort_by_key(|row| core::cmp::Reverse(row.1));
                for (kind, nanos, extents) in rows.iter().take(14) {
                    eprintln!("op {kind} {:?} {:?}", extents, Duration::from_nanos(*nanos));
                }
                let total: u64 = rows.iter().map(|row| row.1).sum();
                eprintln!("op gpu total {:?} over {} ops", Duration::from_nanos(total), rows.len());
                Some(evaluated)
            } else {
                None
            };
            #[cfg(not(feature = "instrument"))]
            let profiled = None::<_>;
            match profiled {
                Some(evaluated) => evaluated,
                None => denoiser
                    .evaluate(named, &extra)
                    .map_err(|why| format!("step {}: {why}", step + 1))?,
            }
            };
            let elapsed = ran.elapsed();
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
                        let path = trace_dir.join(super::oracle::file("traces", name)?);
                        if !path.exists() {
                            eprintln!("trace {name}: no reference at {}", path.display());
                            continue;
                        }
                        let reference = dump_values!(&path);
                        let actual = evaluated
                            .get(*node)
                            .ok_or_else(|| format!("trace {name} is not in the evaluated set"))?
                            .0;
                        let (rms, cosine) = if *name == "vec" {
                            stats!(actual, &reference)
                        } else if name.starts_with("b_imgq") || name.starts_with("b_imgk") || name.starts_with("b_imgv") {
                            let mut token_major = vec![0.0; reference.len()];
                            for token in 0..4096 {
                                for head in 0..24 {
                                    for axis in 0..128 {
                                        token_major[token * 3072 + head * 128 + axis] =
                                            reference[axis + 128 * (head + 24 * token)];
                                    }
                                }
                            }
                            stats!(actual, &token_major)
                        } else {
                            let channels = if *name == "final" { 128 } else { 3072 };
                            let tokens = actual.len() / channels;
                            stats!(actual, &from_channel_major(&reference, tokens, channels))
                        };
                        eprintln!("trace {name}: rms_rel={rms:.6} cosine={cosine:.8}");
                    }
                }
                let (rms, cosine) = stats!(&velocity_values, &reference_velocity);
                eprintln!("velocity1: rms_rel={rms:.6} cosine={cosine:.8} in {elapsed:?}");
                assert!(rms <= 0.101, "velocity rms_rel {rms}");
                assert!(cosine >= 0.999, "velocity cosine {cosine}");
                verdicts += 1;
            }
            euler_step(
                &mut latent_values,
                &velocity_values,
                sigmas[step],
                sigmas[step + 1],
                layout,
            )
            .map_err(|why| why.to_string())?;
            let reference =
                token_major(&named_values!(fixtures.reference, &format!("step{}", step + 1)));
            let (rms, cosine) = stats!(&latent_values, &reference);
            let bar = [0.003, 0.007, 0.016, 0.078][step];
            eprintln!("step{}: rms_rel={rms:.6} cosine={cosine:.8} bar={bar} in {elapsed:?}", step + 1);
            assert!(rms <= bar, "step {} rms_rel {rms} above {bar}", step + 1);
            assert!(cosine >= 0.999, "step {} cosine {cosine}", step + 1);
            verdicts += 1;
        }
        eprintln!("{verdicts} verdicts printed");
        assert_eq!(
            denoiser.plan_counts(),
            (STEPS as usize - 1, 1),
            "the four steps are one plan miss, then three hits on the retained plan"
        );
        Ok(())
    }
}

#[cfg(feature = "std")]
mod dit_shape {
    use proxima_model_interop::diffusion::{
        DitConfig, DitInputs, Graph, attention_scale, build_dit, head_norm_count,
    };

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
        let activation = super::activation::dit().expect("the backbone's activation dtype resolves");
        let mut graph = Graph::new(activation).expect("diffusion settings validate");
        let latent = graph.host("latent", &[IMAGE_TOKENS as u32, 128], graph.unbound());
        let text = graph.host("text", &[TEXT_TOKENS as u32, 7680], graph.unbound());
        let timestep = graph.host("timestep", &[256], graph.unbound());
        let rope_cos = graph.host("rope_cos", &[JOINT as u32, 64], graph.unbound());
        let rope_sin = graph.host("rope_sin", &[JOINT as u32, 64], graph.unbound());
        let config = dit_config();
        let norm_count = head_norm_count(&mut graph, &config);
        let scale = attention_scale(&mut graph, &config)
            .expect("the attention scale builds in the program");
        let inputs = DitInputs {
            latent,
            text,
            timestep,
            rope_cos,
            rope_sin,
            norm_count,
            attention_scale: scale,
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
                for (index, node) in graph.program.iter().enumerate().take(at + 2).skip(at.saturating_sub(14)) {
                    eprintln!("node {index}: {node:?}");
                }
                panic!("{why}");
            }
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod vae {
    use std::env;
    use std::ops::{Add, Mul};
    use std::path::PathBuf;
    use std::time::Instant;

    use conflaguration::Settings;
    use proxima_model_interop::ServingSettings;
    use proxima_model_interop::diffusion::{
        Checkpoint, Decoder, DiffusionSettings, Graph, VaeConfig, build_vae_decode,
    };
    use proxima_numeric::PackedRows;

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

    fn config(window: u32) -> Result<VaeConfig, String> {
        Ok(VaeConfig {
            window,
            channels: CHANNELS as u32,
            block_channels: [128, 256, 512, 512],
            layers_per_block: 2,
            groups: 32,
            out_channels: 3,
            height: SIDE as u32,
            width: SIDE as u32,
            // `PROXIMA_DIFFUSION_CONV_TAPS=nine` A/Bs the lowering without a
            // rebuild; unset keeps the default three-tap form.
            conv_taps: DiffusionSettings::try_resolved()
                .map_err(|why| why.to_string())?
                .conv_taps(),
        })
    }

    /// `step4` is the 128-channel latent in channel-major order; the decoder
    /// wants `[row, column, channels / patch^2]` after the checkpoint's own
    /// per-channel de-normalisation and the 2x2 pixel shuffle.
    fn decoder_input<T>(raw: &[T], mean: &[T], deviation: &[T]) -> Result<Vec<T>, String>
    where
        T: Copy + Default + Mul<Output = T> + Add<Output = T>,
    {
        if raw.len() != LATENT * LATENT * LATENT_CHANNELS {
            return Err(format!("latent has {} values", raw.len()));
        }
        let mut values = vec![T::default(); TOKENS * CHANNELS];
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

    fn serving_settings() -> Result<ServingSettings, String> {
        ServingSettings::from_env().map_err(|why| why.to_string())
    }

    #[test]
    fn the_decoder_reproduces_the_reference_image() -> Outcome {
        let reference = reference_dir()?;
        let model = model_dir()?;
        let checkpoint = Checkpoint::open(&model.join("comfy/flux2-vae.safetensors"))?;
        let (_, mean_rows) = checkpoint.values("bn.running_mean", None, None, false)?;
        let (_, variance_rows) = checkpoint.values("bn.running_var", None, None, false)?;
        let mut mean = Vec::new();
        mean_rows.decode_all(&mut mean).map_err(|why| why.to_string())?;
        let mut variance = Vec::new();
        variance_rows.decode_all(&mut variance).map_err(|why| why.to_string())?;
        let deviation: Vec<_> = variance.iter().map(|value| value.sqrt()).collect();
        let raw = named_values!(reference, "step4");
        let latent = decoder_input(&raw, &mean, &deviation)?;

        let mut graph = Graph::new(super::activation::vae()?).expect("diffusion settings validate");
        let input = graph.host(
            "latent",
            &[TOKENS as u32, CHANNELS as u32, 1],
            graph.unbound(),
        );
        let started = Instant::now();
        let window = checkpoint.window_extent("decoder.conv_in.weight")?;
        let output =
            build_vae_decode(&mut graph, &config(window)?, input).map_err(|why| why.to_string())?;
        let image = output.image;
        eprintln!(
            "decoder: {} ops, {} weights, {} hosts built in {:?}",
            graph.program.len(),
            graph.weights.len(),
            graph.hosts.len(),
            started.elapsed()
        );
        let loading = Instant::now();
        let activation = graph.activation();
        let mut decoder = Decoder::new(
            &checkpoint,
            graph,
            image,
            &[TOKENS as u64],
            DiffusionSettings::try_resolved()
                .map_err(|why| why.to_string())?
                .vae_weights,
            &serving_settings()?.as_serving_config(&[]),
        )?;
        eprintln!("vae weights loaded in {:?}", loading.elapsed());

        let mut call_graph = Graph::new(activation).expect("diffusion settings validate");
        call_graph.host("latent", &[TOKENS as u32, CHANNELS as u32, 1], call_graph.unbound());
        let latent_rows = PackedRows::from_values(call_graph.activation_format(), &latent)
            .map_err(|why| why.to_string())?;
        let ran = Instant::now();
        let blocks = call_graph.host_blocks(&[("latent", &latent_rows)])?;
        let evaluated = decoder
            .evaluate(&blocks)
            .map_err(|why| format!("decode: {why}"))?;
        let elapsed = ran.elapsed();
        let decoded = evaluated.get(image).ok_or("image is not an output")?.0;
        eprintln!("decoded [{}] in {:?}", decoded.len(), elapsed);

        let scaled: Vec<_> = decoded
            .iter()
            .map(|value| ((value + 1.0) * 0.5).clamp(0.0, 1.0))
            .collect();
        let decoded = scaled.as_slice();
        let channel_major = named_values!(reference, "vae_out");
        let tokens = channel_major.len() / 3;
        let token_major = super::token_major_and(&channel_major, tokens, 3);
        let ours = summary!(decoded);
        let theirs = summary!(&token_major);
        eprintln!(
            "ours min {:.3} max {:.3} mean {:.3}; reference min {:.3} max {:.3} mean {:.3}",
            ours.0, ours.1, ours.2, theirs.0, theirs.1, theirs.2
        );
        for shift_y in [-1i32, 0, 1] {
            for shift_x in [-1i32, 0, 1] {
                let mut shifted = vec![0.0; decoded.len()];
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
                let (moved, moved_cos) = fidelity!(&shifted, &token_major);
                if (shift_y, shift_x) != (0, 0) {
                    eprintln!("shift ({shift_y},{shift_x}): psnr={moved:.2} cos={moved_cos:.6}");
                }
            }
        }
        let (signal, cosine) = fidelity!(decoded, &token_major);
        for channel in 0..3usize {
            let ours: Vec<_> = (0..token_major.len() / 3).map(|token| decoded[token * 3 + channel]).collect();
            let theirs: Vec<_> = (0..token_major.len() / 3).map(|token| token_major[token * 3 + channel]).collect();
            let (value, channel_cos) = fidelity!(&ours, &theirs);
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
    use proxima_model_interop::diffusion::{DiffusionSettings, Graph, VaeConfig, build_vae_decode};

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
            conv_taps: DiffusionSettings::try_resolved()
                .expect("a PROXIMA_DIFFUSION_ override must parse and validate")
                .conv_taps(),
            window: super::shape_only_window(),
        };
        let activation = super::activation::vae().expect("the decoder's activation dtype resolves");
        let mut graph = Graph::new(activation).expect("diffusion settings validate");
        let latent = graph.host("latent", &[16384, 32, 1], graph.unbound());
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
                for (index, node) in graph.program.iter().enumerate().take(at + 2).skip(at.saturating_sub(6)) {
                    eprintln!("node {index}: {node:?}");
                }
                panic!("{why}");
            }
        }
    }
}

#[cfg(feature = "std")]
mod vae_bind {
    use proxima_model_interop::diffusion::{DiffusionSettings, Graph, VaeConfig, build_vae_decode};
    use proxima_numeric::NumericPolicy;

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
            conv_taps: DiffusionSettings::try_resolved()
                .expect("a PROXIMA_DIFFUSION_ override must parse and validate")
                .conv_taps(),
            window: super::shape_only_window(),
        };
        let activation = super::activation::vae().expect("the decoder's activation dtype resolves");
        let mut graph = Graph::new(activation).expect("diffusion settings validate");
        let latent = graph.host("latent", &[16384, 32, 1], graph.unbound());
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
                for (index, node) in graph.program.iter().enumerate().take(at + 2).skip(at.saturating_sub(6)) {
                    eprintln!("node {index}: {node:?}");
                }
                panic!("{why}");
            }
        };
        for (index, node) in graph.program.iter().enumerate().take(57).skip(50) {
            eprintln!("p{index}: {node:?}");
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
    use conflaguration::Settings;
    use proxima_model_interop::ServingSettings;
    use proxima_model_interop::diffusion::{Checkpoint, Decoder, DiffusionSettings, Graph};
    use proxima_numeric::PackedRows;
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
        let mut graph = Graph::new(super::activation::vae()?).expect("diffusion settings validate");
        let rows = SIDE * SIDE;
        let latent = graph.host("latent", &[rows, GROUPS, 1], graph.unbound());
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
            proxima_tensor::spec::window_pad_index(SIDE, SIDE, 3),
        );
        let mask: Vec<_> = proxima_tensor::spec::window_pad_mask(SIDE, SIDE, 3)
            .into_iter()
            .map(|value| value as _)
            .collect();
        let mask_rows = PackedRows::from_values(graph.activation_format(), &mask);
        let pad_mask = graph.host(
            "pad.mask",
            &[proxima_tensor::spec::window_pad_rows(SIDE, SIDE, 3)],
            mask_rows,
        );
        let crop_index = graph.index_leaf(
            "crop.index",
            proxima_tensor::spec::window_crop_index(SIDE, SIDE, 3),
        );
        let convolved = proxima_tensor::spec::window_same(
            &mut graph.program,
            latent,
            &taps,
            pad_index,
            pad_mask,
            crop_index,
            SIDE,
            SIDE,
            3,
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
        let latent_values = vec![0.5; (rows * GROUPS) as usize];
        let mut call_graph = Graph::new(graph.activation()).expect("diffusion settings validate");
        call_graph.host("latent", &[rows, GROUPS, 1], call_graph.unbound());
        let latent_rows = PackedRows::from_values(call_graph.activation_format(), &latent_values)
            .map_err(|why| why.to_string())?;
        let mut decoder = Decoder::new(
            &checkpoint,
            graph,
            image,
            &[rows as u64],
            DiffusionSettings::try_resolved()
                .map_err(|why| why.to_string())?
                .vae_weights,
            &ServingSettings::from_env()
                .map_err(|why| why.to_string())?
                .as_serving_config(&[]),
        )?;
        let blocks = call_graph.host_blocks(&[("latent", &latent_rows)])?;
        let evaluated = decoder
            .evaluate(&blocks)
            .map_err(|why| format!("run: {why}"))?;
        let values = evaluated.get(image).ok_or("no image")?.0;
        let rms = measured!(::proxima_numeric::compare::rms(&host_rows!(values)))[0];
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

    use proxima_model_interop::diffusion::{DiffusionSettings, Graph};
    use proxima_numeric::{NumericError, NumericFormat, PackedRows, Philox};
    use proxima_tensor::QuantizedBlock;
    use proxima_tensor::spec::{
        window_crop_index, window_pad_index, window_pad_mask, window_pad_rows,
        window_same_one_reduce, window_taps_index,
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

    fn values(format: &NumericFormat, count: usize, seed: u64) -> Result<PackedRows, NumericError> {
        PackedRows::from_values(format, &Philox::new(seed).normal(count))
    }

    struct Case {
        program: Vec<proxima_tensor::op::Op>,
        blocks: Vec<(&'static str, PackedRows)>,
        root: proxima_tensor::op::NodeId,
        want: PackedRows,
    }

    /// Host-element rows of an index or mask table, cast to the host element.
    macro_rules! table_rows {
        ($table:expr) => {{
            let cast: Vec<_> = $table.into_iter().map(|value| value as _).collect();
            host_rows!(&cast)
        }};
    }

    /// The program, its bound values, and the direct-loop answer.
    fn case() -> Result<Case, Box<dyn std::error::Error>> {
        let rows = HEIGHT * WIDTH;
        let mut graph = Graph::new(super::activation::vae()?)?;
        let image_rows = values(graph.activation_format(), (rows * IN_G * IN_C) as usize, 7);
        let activation = graph.host("image", &[rows, IN_G, IN_C], image_rows);
        let pad_index = graph.index_leaf("pad.index", window_pad_index(HEIGHT, WIDTH, 3));
        let mask: Vec<_> = window_pad_mask(HEIGHT, WIDTH, 3)
            .into_iter()
            .map(|value| value as _)
            .collect();
        let mask_rows = PackedRows::from_values(graph.activation_format(), &mask);
        let pad_mask = graph.host("pad.mask", &[window_pad_rows(HEIGHT, WIDTH, 3)], mask_rows);
        let crop_index = graph.index_leaf("crop.index", window_crop_index(HEIGHT, WIDTH, 3));
        let columns_index = graph.index_leaf("columns.index", window_taps_index(HEIGHT, WIDTH, 3));
        let kernel_rows = values(
            graph.activation_format(),
            (OUT_G * OUT_C * 9 * IN_G * IN_C) as usize,
            31,
        );
        let kernel = graph.host("kernel", &[OUT_G, OUT_C, 9, IN_G, IN_C], kernel_rows);
        let root = window_same_one_reduce(
            &mut graph.program,
            activation,
            kernel,
            pad_index,
            pad_mask,
            crop_index,
            columns_index,
            3,
        )?;

        // The definition, on the same values.
        let image_host = graph_host(&graph, "image")?;
        let weights_host = graph_host(&graph, "kernel")?;
        let image = decoded!(image_host);
        let weights = decoded!(weights_host);
        let mut want = vec![0.0; (rows * OUT_G * OUT_C) as usize];
        for row in 0..HEIGHT as usize {
            for column in 0..WIDTH as usize {
                for og in 0..OUT_G as usize {
                    for oc in 0..OUT_C as usize {
                        let mut sum = 0.0;
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

        let image_rows = PackedRows::from_values(graph.activation_format(), &image)?;
        let held = graph.host_blocks(&[("image", &image_rows)])?;
        assert!(!held.is_empty());
        let blocks = vec![
            ("image", image_host),
            ("kernel", weights_host),
            ("pad.mask", table_rows!(window_pad_mask(HEIGHT, WIDTH, 3))),
            ("pad.index", table_rows!(window_pad_index(HEIGHT, WIDTH, 3))),
            ("crop.index", table_rows!(window_crop_index(HEIGHT, WIDTH, 3))),
            ("columns.index", table_rows!(window_taps_index(HEIGHT, WIDTH, 3))),
        ];
        Ok(Case {
            program: graph.program,
            blocks,
            root,
            want: host_rows!(&want),
        })
    }

    fn graph_host(graph: &Graph, name: &str) -> Result<PackedRows, String> {
        let leaf = graph
            .hosts
            .iter()
            .find(|leaf| leaf.name == name)
            .ok_or_else(|| format!("no host leaf named {name}"))?;
        leaf.values.to_host_element().map_err(|why| why.to_string())
    }

    fn quantized<'a, T>(
        blocks: &'a [(&'static str, Vec<T>)],
    ) -> Vec<(&'static str, QuantizedBlock<'a>)>
    where
        QuantizedBlock<'a>: From<&'a [T]>,
    {
        blocks
            .iter()
            .map(|(name, values)| (*name, QuantizedBlock::from(values.as_slice())))
            .collect()
    }

    macro_rules! decoded_blocks {
        ($blocks:expr) => {{
            let mut out = Vec::new();
            for (name, rows) in &$blocks {
                out.push((*name, decoded!(rows)));
            }
            out
        }};
    }

    #[test]
    fn a_split_contraction_axis_reduce_agrees_between_cpu_and_metal() -> Result<(), String> {
        let case = case().expect("the three-block convolution case builds");
        let decoded_case = decoded_blocks!(case.blocks);
        let bound = quantized(&decoded_case);
        let want = decoded!(case.want);

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
        // Relative: the contract is a 4608-term dot product here, so an
        // absolute micro-bound would be measuring the accumulator's rounding
        // rather than the lowering.
        let scale = want
            .iter()
            .map(|value| value.abs())
            .reduce(|worst, value| worst.max(value))
            .unwrap_or_default();
        let worst_cpu = cpu
            .iter()
            .zip(&want)
            .map(|(cpu, want)| (cpu - want).abs())
            .reduce(|worst, value| worst.max(value))
            .unwrap_or_default()
            / scale;
        assert!(
            worst_cpu < 1e-4,
            "the CPU form differs from the definition by {worst_cpu} relative"
        );

        // Metal on its own thread: `waitUntilCompleted` never returns under the
        // defect, so the guard is what turns that into a failure.
        let program = case.program.clone();
        let root = case.root;
        let blocks = decoded_case.clone();
        let codec = DiffusionSettings::try_resolved()
            .map_err(|why| why.to_string())?
            .vae_weights;
        let (sender, receiver) = mpsc::channel();
        std::thread::spawn(move || {
            // The kernel is bound exactly as the decoder binds it: packed
            // packed weight rows, not host elements. The host binding passes; the packed
            // one is what the served graph actually uses.
            let packed = PackedRows::from_values(
                &codec,
                &blocks.iter().find(|(name, _)| *name == "kernel").expect("kernel").1,
            )
            .expect("the kernel encodes in the bound weight format");
            let mut bound: Vec<(&'static str, QuantizedBlock<'_>)> = Vec::new();
            for (name, values) in &blocks {
                bound.push((
                    *name,
                    if *name == "kernel" {
                        QuantizedBlock::Packed {
                            codec,
                            bytes: packed.as_bytes(),
                        }
                    } else {
                        QuantizedBlock::from(values.as_slice())
                    },
                ));
            }
            // a raw program, not a stage: no crate-public way runs one on a
            // `BackendRuntime`, so this reproduction keeps its own plan
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
        let nanos = metal_started.elapsed().as_nanos().max(1);
        let macs = u128::from(HEIGHT * WIDTH)
            * u128::from(OUT_G * OUT_C)
            * 9
            * u128::from(IN_G * IN_C);
        eprintln!(
            "split_reduce metal {nanos} ns for {macs} MACs = {} MACs per microsecond",
            macs * 1000 / nanos
        );
        assert_eq!(metal.len(), cpu.len(), "shape");
        let worst = metal
            .iter()
            .zip(&cpu)
            .map(|(metal, cpu)| (metal - cpu).abs())
            .reduce(|worst, value| worst.max(value))
            .unwrap_or_default();
        // The kernel is bound in the weight codec, so the agreement bound is that codec's
        // rounding accumulated over the contraction, not the graph's own.
        assert!(
            worst / scale < 2e-2,
            "Metal differs from the CPU by {} relative",
            worst / scale
        );
        Ok(())
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod vae_prefix {
    use std::env;
    use std::path::PathBuf;

    use conflaguration::Settings;
    use proxima_model_interop::ServingSettings;
    use proxima_model_interop::diffusion::{
        Checkpoint, Decoder, DiffusionSettings, Graph, VaeConfig, build_vae_decode,
    };
    use proxima_numeric::PackedRows;

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
            window: checkpoint.window_extent("decoder.conv_in.weight")?,
            conv_taps: DiffusionSettings::try_resolved()
                .map_err(|why| why.to_string())?
                .conv_taps(),
        };
        let mut graph = Graph::new(super::activation::vae()?).expect("diffusion settings validate");
        let latent = graph.host("latent", &[SIDE * SIDE, 32, 1], graph.unbound());
        let output = build_vae_decode(&mut graph, &config, latent).map_err(|why| why.to_string())?;
        let latent_values = vec![0.1; TOKENS * 32];
        let mut call_graph = Graph::new(graph.activation()).expect("diffusion settings validate");
        call_graph.host("latent", &[SIDE * SIDE, 32, 1], call_graph.unbound());
        let latent_rows = PackedRows::from_values(call_graph.activation_format(), &latent_values)
            .map_err(|why| why.to_string())?;
        let mut decoder = Decoder::new(
            &checkpoint,
            graph,
            output.image,
            &[TOKENS as u64],
            DiffusionSettings::try_resolved()
                .map_err(|why| why.to_string())?
                .vae_weights,
            &ServingSettings::from_env()
                .map_err(|why| why.to_string())?
                .as_serving_config(&[]),
        )?;
        let blocks = call_graph.host_blocks(&[("latent", &latent_rows)])?;
        let mut faulted = false;
        for (name, node) in &output.trace {
            if let Err(why) = decoder.evaluate_nodes(&blocks, &[*node]) {
                eprintln!("stage {name} FAULTS: {why}");
                faulted = true;
                break;
            }
            eprintln!("stage {name} ok");
        }
        if !faulted {
            eprintln!("every decoder stage ran");
        }
        Ok(())
    }
}

#[cfg(feature = "std")]
mod vae_gathers {
    use proxima_model_interop::diffusion::{DiffusionSettings, Graph, VaeConfig, build_vae_decode};
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
            conv_taps: DiffusionSettings::try_resolved()
                .expect("a PROXIMA_DIFFUSION_ override must parse and validate")
                .conv_taps(),
            window: super::shape_only_window(),
        };
        let activation = super::activation::vae().expect("the decoder's activation dtype resolves");
        let mut graph = Graph::new(activation).expect("diffusion settings validate");
        let latent = graph.host("latent", &[256, 32, 1], graph.unbound());
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
                let mut decoded = Vec::new();
                host.values
                    .decode_all(&mut decoded)
                    .expect("a gather's leaf decodes");
                let maximum = decoded
                    .iter()
                    .copied()
                    .reduce(|acc, value| acc.max(value))
                    .unwrap_or_default();
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
