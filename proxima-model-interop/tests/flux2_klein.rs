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
