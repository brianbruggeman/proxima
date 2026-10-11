use std::collections::BTreeMap;
use std::env;
use std::fs;

fn field(line: &str, name: &str) -> Option<String> {
    let marker = format!(" {name}=");
    let at = line.find(&marker)?;
    let rest = &line[at + marker.len()..];
    if rest.starts_with('[') {
        return Some(rest[..=rest.find(']')?].to_string());
    }
    Some(rest[..rest.find(' ').unwrap_or(rest.len())].to_string())
}

fn number(line: &str, name: &str) -> Option<f64> {
    field(line, name)?.parse().ok()
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).expect("finite"));
    sorted[sorted.len() / 2]
}

fn cov_percent(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / (values.len() as f64 - 1.0);
    100.0 * variance.sqrt() / mean
}

fn join(values: &[f64], precision: usize) -> String {
    values.iter().map(|value| format!("{value:.precision$}")).collect::<Vec<_>>().join(" / ")
}

struct Cell {
    base: Vec<f64>,
    arms: BTreeMap<String, Vec<f64>>,
    bits: BTreeMap<String, String>,
    count: String,
}

fn arm_tag(arm: &str, sha: &str) -> String {
    match arm.strip_prefix(sha) {
        Some(tail) => tail.trim_start_matches('.').to_string(),
        None => arm.to_string(),
    }
}

fn main() {
    let arguments: Vec<String> = env::args().collect();
    let value_key = &arguments[1];
    let files = &arguments[2..];
    if value_key == "stepall" {
        step_table(files);
        return;
    }
    let mut cells: BTreeMap<(String, String), Cell> = BTreeMap::new();
    for (process, path) in files.iter().enumerate() {
        for line in fs::read_to_string(path).expect("read").lines() {
            let (Some(sha), Some(extents), Some(arm)) = (field(line, "sha"), field(line, "extents"), field(line, "arm")) else {
                continue;
            };
            let key = (sha[..8].to_string(), extents);
            let tag = arm_tag(&arm, &sha);
            let cell = cells.entry(key).or_insert_with(|| Cell { base: vec![0.0; files.len()], arms: BTreeMap::new(), bits: BTreeMap::new(), count: String::new() });
            if line.starts_with("ab group ") {
                cell.count = field(line, "count").unwrap_or_default();
                if let Some(value) = number(line, value_key) {
                    if tag == "base" {
                        cell.base[process] = value;
                    } else {
                        cell.arms.entry(tag).or_insert_with(|| vec![0.0; files.len()])[process] = value;
                    }
                }
            } else if line.starts_with("ab bits ") && process == 0 {
                cell.bits.insert(tag, format!("{}/{} max_abs={}", field(line, "differing").unwrap_or_default(), field(line, "elements").unwrap_or_default(), field(line, "max_abs").unwrap_or_default()));
            }
        }
    }
    println!("value `{value_key}`; per-process ratio arm/base (interleaved in one process), median [CoV% of the ratios] (min-max); base and arm absolute us per process");
    println!("| group | extents | dispatches | arm | ratio median [CoV%] (min-max) | delta us (median ratio x base median) | base us per process | arm us per process | bits vs base (process 1) |");
    println!("|---|---|---|---|---|---|---|---|---|");
    for ((sha, extents), cell) in &cells {
        for (tag, values) in &cell.arms {
            let ratios: Vec<f64> = values.iter().zip(&cell.base).map(|(arm, base)| arm / base).collect();
            let spread = format!("{:.3}-{:.3}", ratios.iter().cloned().fold(f64::MAX, f64::min), ratios.iter().cloned().fold(f64::MIN, f64::max));
            let delta = (median(&ratios) - 1.0) * median(&cell.base);
            println!(
                "| {sha} | {extents} | {} | {tag} | {:.3} [{:.2}] ({spread}) | {delta:+.3} | {} | {} | {} |",
                cell.count,
                median(&ratios),
                cov_percent(&ratios),
                join(&cell.base, 3),
                join(values, 3),
                cell.bits.get(tag).cloned().unwrap_or_default()
            );
        }
    }
}

fn step_table(files: &[String]) {
    let mut arms: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut order: Vec<String> = Vec::new();
    for (process, path) in files.iter().enumerate() {
        for line in fs::read_to_string(path).expect("read").lines() {
            if !line.starts_with("ab stepall arm=") {
                continue;
            }
            let (Some(arm), Some(step)) = (field(line, "arm"), number(line, "step_ms")) else {
                continue;
            };
            if process == 0 {
                order.push(arm.clone());
            }
            arms.entry(arm).or_insert_with(|| vec![0.0; files.len()])[process] = step;
        }
    }
    let base = arms.get("base").expect("base arm").clone();
    println!("| arm | step ms per process | ratio to base median [CoV%] (min-max) | delta ms |");
    println!("|---|---|---|---|");
    for arm in &order {
        let values = &arms[arm];
        let ratios: Vec<f64> = values.iter().zip(&base).map(|(value, reference)| value / reference).collect();
        let spread = format!("{:.4}-{:.4}", ratios.iter().cloned().fold(f64::MAX, f64::min), ratios.iter().cloned().fold(f64::MIN, f64::max));
        println!("| {arm} | {} | {:.4} [{:.2}] ({spread}) | {:+.3} |", join(values, 4), median(&ratios), cov_percent(&ratios), (median(&ratios) - 1.0) * median(&base));
    }
}
