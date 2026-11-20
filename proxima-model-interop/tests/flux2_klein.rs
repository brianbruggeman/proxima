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
