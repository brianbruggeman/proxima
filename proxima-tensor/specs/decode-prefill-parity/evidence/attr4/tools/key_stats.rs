use std::env;
use std::fs;

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).expect("finite"));
    sorted[sorted.len() / 2]
}

fn main() {
    let arguments: Vec<String> = env::args().collect();
    let path = &arguments[1];
    let line_filter = &arguments[2];
    let skip: usize = arguments[3].parse().expect("leading lines to skip");
    let text = fs::read_to_string(path).expect("read file");
    let lines: Vec<&str> = text.lines().filter(|line| line.contains(line_filter.as_str())).skip(skip).collect();
    println!("{path}: {} lines match `{line_filter}` after skipping {skip}", lines.len());
    for key in &arguments[4..] {
        let marker = format!(" {key}=");
        let values: Vec<f64> = lines
            .iter()
            .filter_map(|line| {
                let at = line.find(&marker)?;
                let rest = &line[at + marker.len()..];
                rest[..rest.find(' ').unwrap_or(rest.len())].parse().ok()
            })
            .collect();
        assert!(!values.is_empty(), "N==0 for key {key}");
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / (values.len() as f64 - 1.0).max(1.0);
        println!(
            "  {key}: n={} median={:.3} cov_pct={:.2} min={:.3} max={:.3}",
            values.len(),
            median(&values),
            100.0 * variance.sqrt() / mean,
            values.iter().copied().fold(f64::INFINITY, f64::min),
            values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        );
    }
}
