use std::collections::BTreeMap;
use std::env;
use std::fs;

const METRICS: [&str; 11] = [
    "prefill_ms",
    "ttft_ms",
    "wall_ms",
    "ms_per_token",
    "cpu_pct",
    "peak_rss_bytes",
    "peak_footprint_bytes",
    "steady_rss_bytes",
    "steady_footprint_bytes",
    "peak_gpu_bytes",
    "steady_gpu_bytes",
];

fn field(line: &str, name: &str) -> Option<String> {
    let marker = format!(" {name}=");
    let at = line.find(&marker)?;
    let rest = &line[at + marker.len()..];
    Some(rest[..rest.find(' ').unwrap_or(rest.len())].to_string())
}

fn cell(line: &str) -> String {
    if let Some(median) = field(line, "median_kept") {
        let kept = field(line, "n_kept").unwrap_or_default();
        let all = field(line, "n_all").unwrap_or_default();
        let cov_kept = field(line, "cov_kept_pct").unwrap_or_default();
        let cov_all = field(line, "cov_all_pct").unwrap_or_default();
        let low = field(line, "min_kept").unwrap_or_default();
        let high = field(line, "max_kept").unwrap_or_default();
        return format!("{} ({cov_all}/{cov_kept}; {low}-{high}; {kept} of {all})", short(&median));
    }
    let median = field(line, "median").unwrap_or_default();
    let maximum = field(line, "max").unwrap_or_default();
    let count = field(line, "n").unwrap_or_default();
    format!("{} (max {}; n {count})", short(&median), short(&maximum))
}

fn short(text: &str) -> String {
    match text.parse::<f64>() {
        Ok(value) if value.abs() >= 1.0e6 => format!("{:.1} MB", value / 1.0e6),
        Ok(value) => format!("{value:.3}"),
        Err(_) => text.to_string(),
    }
}

fn main() {
    let path = env::args().nth(1).expect("decode_arms.out path");
    let text = fs::read_to_string(path).expect("read decode_arms.out");
    let mut table: BTreeMap<(String, String), BTreeMap<String, String>> = BTreeMap::new();
    for line in text.lines().filter(|line| line.starts_with("summary arm=")) {
        let arm = field(&format!(" {line}"), "arm").expect("arm");
        let metric = field(&format!(" {line}"), "metric").expect("metric");
        let (model, label) = arm.split_once('.').expect("model.arm");
        if METRICS.contains(&metric.as_str()) {
            table
                .entry((model.to_string(), label.to_string()))
                .or_default()
                .insert(metric, cell(line));
        }
    }
    assert!(!table.is_empty(), "N==0: no summary lines parsed");
    println!("cell: `median (CoV all / CoV kept; min-max kept; n kept of n all)`; byte metrics in MB (decimal)");
    println!();
    println!("| model | arm | {} |", METRICS.join(" | "));
    println!("|---|---|{}|", METRICS.iter().map(|_| "---").collect::<Vec<_>>().join("|"));
    for ((model, label), metrics) in &table {
        let cells: Vec<String> = METRICS
            .iter()
            .map(|name| metrics.get(*name).cloned().unwrap_or_else(|| "MISSING".to_string()))
            .collect();
        println!("| {model} | {label} | {} |", cells.join(" | "));
    }
}
