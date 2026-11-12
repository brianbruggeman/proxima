//! the arithmetic `bge_eval.rs` and the 2026-09 harness share: population-variance `CoV`, `CLS` `l2` norm, cosine.

#[must_use]
pub fn mean(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().sum::<f64>() / count_of(samples)
}

fn count_of(samples: &[f64]) -> f64 {
    f64::from(u32::try_from(samples.len()).unwrap_or(u32::MAX))
}

/// `sqrt(population variance) / mean`, the formula `bge_eval.rs::coefficient_of_variation` uses.
#[must_use]
pub fn coefficient_of_variation(samples: &[f64], mean_value: f64) -> f64 {
    if samples.len() < 2 || mean_value == 0.0 {
        return 0.0;
    }
    let variance = samples.iter().map(|value| (value - mean_value).powi(2)).sum::<f64>() / count_of(samples);
    variance.sqrt() / mean_value
}

#[must_use]
pub fn l2_normalize(vector: &[f32]) -> Vec<f32> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    vector.iter().map(|value| value / norm).collect()
}

#[must_use]
pub fn cosine(left: &[f32], right: &[f32]) -> f64 {
    let dot: f64 = left.iter().zip(right).map(|(a_val, b_val)| f64::from(*a_val) * f64::from(*b_val)).sum();
    let left_norm = left.iter().map(|value| f64::from(*value).powi(2)).sum::<f64>().sqrt();
    let right_norm = right.iter().map(|value| f64::from(*value).powi(2)).sum::<f64>().sqrt();
    dot / (left_norm * right_norm)
}

#[must_use]
pub fn max_abs_diff(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a_val, b_val)| (a_val - b_val).abs()).fold(0.0, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cov_of_identical_samples_is_zero() {
        assert!(coefficient_of_variation(&[5.0, 5.0, 5.0], 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn cov_uses_population_variance() {
        let samples = [4.0, 6.0];
        let cov = coefficient_of_variation(&samples, mean(&samples));
        assert!((cov - 0.2).abs() < 1e-12, "{cov}");
    }

    #[test]
    fn l2_normalize_yields_unit_norm_and_self_cosine_one() {
        let unit = l2_normalize(&[3.0, 4.0, 0.0]);
        assert!((unit[0] - 0.6).abs() < 1e-7 && (unit[1] - 0.8).abs() < 1e-7);
        assert!((cosine(&unit, &unit) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn max_abs_diff_reports_the_largest_component_gap() {
        assert!((max_abs_diff(&[1.0, 2.0, 3.0], &[1.0, 2.5, 2.0]) - 1.0).abs() < f32::EPSILON);
    }
}
