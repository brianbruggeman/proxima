use std::collections::BTreeMap;
use std::env;
use std::fs;

struct Sample {
    cost_ms: f64,
    base_ms: f64,
}

struct Key {
    entry: String,
    sha: String,
    threads: String,
    extents: String,
    count: usize,
}

fn field<'a>(line: &'a str, name: &str) -> &'a str {
    let marker = format!("{name}=");
    let start = line.find(&marker).map(|at| at + marker.len()).unwrap_or(0);
    let rest = &line[start..];
    if name == "extents" {
        let end = rest.find(']').map(|at| at + 1).unwrap_or(rest.len());
        return &rest[..end];
    }
    let end = rest.find(' ').unwrap_or(rest.len());
    &rest[..end]
}

fn median(values: &mut Vec<f64>) -> f64 {
    values.sort_by(|left, right| left.partial_cmp(right).expect("finite"));
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        values[middle]
    } else {
        (values[middle - 1] + values[middle]) / 2.0
    }
}

fn parse_run(path: &str) -> Vec<(String, Key, Sample)> {
    let text = fs::read_to_string(path).expect("read run file");
    text.lines()
        .filter(|line| line.starts_with("ab omit "))
        .map(|line| {
            let key = Key {
                entry: field(line, "entry").to_string(),
                sha: field(line, "sha").to_string(),
                threads: field(line, "threads").to_string(),
                extents: field(line, "extents").to_string(),
                count: field(line, "count").parse().expect("count"),
            };
            let sample = Sample {
                cost_ms: field(line, "cost_ms").parse().expect("cost_ms"),
                base_ms: field(line, "base_ms").parse().expect("base_ms"),
            };
            let id = format!("{}|{}|{}", key.sha, key.threads, key.extents);
            (id, key, sample)
        })
        .collect()
}

fn main() {
    let paths: Vec<String> = env::args().skip(1).collect();
    assert!(!paths.is_empty(), "N==0: no run files given");
    let mut by_group: BTreeMap<String, (Key, Vec<Sample>)> = BTreeMap::new();
    for path in &paths {
        let parsed = parse_run(path);
        assert!(!parsed.is_empty(), "N==0 omit lines in {path}");
        println!("run file {path}: {} omit lines", parsed.len());
        for (id, key, sample) in parsed {
            by_group.entry(id).or_insert_with(|| (key, Vec::new())).1.push(sample);
        }
    }
    let mut rows: Vec<(f64, String)> = Vec::new();
    let mut base_all: Vec<f64> = Vec::new();
    for (_, (key, samples)) in &by_group {
        let mut costs: Vec<f64> = samples.iter().map(|sample| sample.cost_ms).collect();
        let mut bases: Vec<f64> = samples.iter().map(|sample| sample.base_ms).collect();
        base_all.extend(bases.iter().copied());
        let low = costs.iter().cloned().fold(f64::INFINITY, f64::min);
        let high = costs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let runs = costs.len();
        let cost_median = median(&mut costs);
        let base_median = median(&mut bases);
        let per_dispatch_us = cost_median * 1000.0 / key.count as f64;
        let entry_short: String = key.entry.chars().take(96).collect();
        rows.push((
            cost_median,
            format!(
                "| {entry_short} | {} | {} | {} | {} | {runs} | {cost_median:.4} | {low:.4} to {high:.4} | {per_dispatch_us:.2} | {base_median:.3} |",
                &key.sha[..8], key.threads, key.extents, key.count
            ),
        ));
    }
    rows.sort_by(|left, right| right.0.partial_cmp(&left.0).expect("finite"));
    let total: f64 = rows.iter().map(|row| row.0).sum();
    println!();
    println!("groups={} runs_per_group_max={} sum_of_median_cost_ms={total:.3} median_base_ms={:.3}", rows.len(), paths.len(), median(&mut base_all));
    println!();
    println!("| rank | entry | sha8 | threads | extents | dispatches | runs | median cost ms | cost range ms | us per dispatch | median base ms |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    for (rank, (_, line)) in rows.iter().enumerate() {
        println!("| {} {line}", rank + 1);
    }
}
